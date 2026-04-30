import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import sparse
from scipy.sparse.linalg import spsolve
from iondynamics.microstructure import MicrostructureGrid, PhaseMap

@dataclass
class TransportSolverConfig:
    """2D輸送ソルバの設定"""
    case_name: str = "default_transport"
    dt: float = 1.0  # [s]
    t_final: float = 3600.0  # [s]
    
    # 境界条件 (separator, current_collector, sides)
    bc_separator: str = "constant_concentration"  # or "constant_flux"
    bc_value_separator: float = 1000.0  # [mol/m3] or [mol/m2/s]
    bc_sides: str = "periodic"  # or "symmetric"
    
    # 物理パラメータ
    d_electrolyte_bulk: float = 2.6e-10  # [m2/s]
    k_electrolyte_bulk: float = 0.5  # [S/m]
    c_initial: float = 1000.0  # [mol/m3]
    applied_current_density: float = 20.0  # [A/m2] (全界面での合計電流密度の目安)
    
    # 反応項の設定
    source_distribution: str = "uniform_interface"  # or "volume_averaged"
    
    # 数値計算設定
    method: str = "implicit"  # or "crank-nicolson"
    save_interval_steps: int = 10
    parallel_n_cpu: int = 1
    use_gpu: bool = False
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class PropertyMap:
    """空間的な輸送特性マップ"""
    deff_map: np.ndarray  # [Nz, Nx] [m2/s]
    keff_map: np.ndarray  # [Nz, Nx] [S/m]
    pore_mask: np.ndarray  # [Nz, Nx] bool
    interface_mask: np.ndarray  # [Nz, Nx] bool (ソース項が適用される位置)
    porosity_map: np.ndarray  # [Nz, Nx] (局所空隙率、通常は0 or 1だが粗視化時用)

@dataclass
class ElectrolyteFieldResult:
    """2D輸送計算の結果"""
    time: np.ndarray  # [Nt]
    c_e: Optional[np.ndarray] = None  # [Nt, Nz, Nx] [mol/m3]
    phi_e: Optional[np.ndarray] = None  # [Nz, Nx] or [Nt, Nz, Nx] [V]
    j_e_x: Optional[np.ndarray] = None  # [Nz, Nx] [A/m2]
    j_e_z: Optional[np.ndarray] = None  # [Nz, Nx] [A/m2]
    
    # 集約データ
    c_e_avg: Optional[np.ndarray] = None  # [Nt, Nz]
    z_coords_um: Optional[np.ndarray] = None
    
    kpis: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SolverPerformanceInfo:
    matrix_size: int
    assembly_time: float
    solve_time_total: float
    n_steps: int
    device: str = "CPU"

class TransportSolver2D:
    """2Dイオン輸送ソルバ本体"""
    
    def __init__(self, config: TransportSolverConfig, grid: MicrostructureGrid):
        self.config = config
        self.grid = grid
        self.prop_map: Optional[PropertyMap] = None
        
    def prepare_property_map(self, phase_map: PhaseMap) -> PropertyMap:
        """PhaseMapから物性マップを生成する"""
        nz, nx = phase_map.data.shape
        
        # 0=pore, 1=active material, 2=CBD
        pore_mask = (phase_map.data == 0)
        active_mask = (phase_map.data == 1)
        
        # 実効物性の割り当て (MVP: pore内は一定、それ以外は微小値)
        # 将来的にBruggemanやCBDの寄与を入れられるようにする
        deff_map = np.where(pore_mask, self.config.d_electrolyte_bulk, 1e-20)
        keff_map = np.where(pore_mask, self.config.k_electrolyte_bulk, 1e-20)
        
        # 界面マスクの作成 (active material に隣接する pore)
        # 簡易的に active_mask の周囲を抽出
        from scipy.ndimage import binary_dilation
        dilated_active = binary_dilation(active_mask)
        interface_mask = dilated_active & pore_mask
        
        self.prop_map = PropertyMap(
            deff_map=deff_map,
            keff_map=keff_map,
            pore_mask=pore_mask,
            interface_mask=interface_mask,
            porosity_map=pore_mask.astype(float)
        )
        return self.prop_map

    def solve_steady_potential(self, progress_callback=None) -> ElectrolyteFieldResult:
        """定常ポテンシャル場を解く"""
        if self.prop_map is None:
            raise ValueError("Property map not prepared. Call prepare_property_map first.")
            
        import time
        t_start = time.time()
        
        nz, nx = self.grid.nz, self.grid.nx
        dz = self.grid.resolution * 1e-6
        dx = self.grid.resolution * 1e-6
        
        n_total = nz * nx
        
        if progress_callback:
            progress_callback({"message": f"Assembling potential matrix ({nz}x{nx})..."})
            
        A = sparse.lil_matrix((n_total, n_total))
        b = np.zeros(n_total)
        
        keff = self.prop_map.keff_map
        interface_mask = self.prop_map.interface_mask
        
        # ソース項の計算 (全電流を界面に分配)
        # applied_current_density [A/m2] は電極投影面積あたりの電流とする
        total_current = self.config.applied_current_density * (nx * dx) # [A/m] (2Dなので奥行き1mあたり)
        n_interface = np.sum(interface_mask)
        if n_interface > 0:
            source_per_pixel = total_current / n_interface
        else:
            source_per_pixel = 0.0
            
        for i in range(nz):
            if progress_callback and i % 10 == 0:
                progress_callback({"message": f"Assembling matrix: row {i}/{nz}"})
            for j in range(nx):
                idx = i * nx + j
                
                # 境界条件: セパレータ側 (z=0)
                if i == 0:
                    A[idx, idx] = 1.0
                    b[idx] = 0.0 # φ_e = 0
                    continue
                
                # 内部接点または集電体側境界
                # ∇·(κ ∇φ) = (κ_i+1/2 (φ_i+1 - φ_i) / dz - κ_i-1/2 (φ_i - φ_i-1) / dz) / dz + ...
                
                # Z方向 (i)
                # i+1 (集電体側)
                if i < nz - 1:
                    k_half = 0.5 * (keff[i, j] + keff[i+1, j])
                    coeff = k_half / dz**2
                    A[idx, idx] -= coeff
                    A[idx, (i+1)*nx + j] += coeff
                else:
                    # 集電体側 (z=max) は Neumann: dφ/dz = 0 -> 断熱
                    pass
                
                # i-1 (セパレータ側)
                k_half_prev = 0.5 * (keff[i, j] + keff[i-1, j])
                coeff_prev = k_half_prev / dz**2
                A[idx, idx] -= coeff_prev
                A[idx, (i-1)*nx + j] += coeff_prev
                
                # X方向 (j)
                # j+1
                j_next = (j + 1) % nx if self.config.bc_sides == "periodic" else min(j + 1, nx - 1)
                if j < nx - 1 or self.config.bc_sides == "periodic":
                    k_half_x = 0.5 * (keff[i, j] + keff[i, j_next])
                    coeff_x = k_half_x / dx**2
                    if not (self.config.bc_sides != "periodic" and j == nx - 1):
                        A[idx, idx] -= coeff_x
                        A[idx, i*nx + j_next] += coeff_x
                
                # j-1
                j_prev = (j - 1) % nx if self.config.bc_sides == "periodic" else max(j - 1, 0)
                if j > 0 or self.config.bc_sides == "periodic":
                    k_half_x_prev = 0.5 * (keff[i, j] + keff[i, j_prev])
                    coeff_x_prev = k_half_x_prev / dx**2
                    if not (self.config.bc_sides != "periodic" and j == 0):
                        A[idx, idx] -= coeff_x_prev
                        A[idx, i*nx + j_prev] += coeff_x_prev

                # ソース項 (A/m3)
                if interface_mask[i, j]:
                    # ピクセル体積(面積)で割る
                    b[idx] = - source_per_pixel / (dx * dz)

        if progress_callback:
            progress_callback({"message": "Solving steady potential..."})
            
        t_assembly = time.time() - t_start
        
        # 解く
        A_csr = A.tocsr()
        phi_flat = spsolve(A_csr, b)
        phi_e = phi_flat.reshape((nz, nx))
        
        t_solve = time.time() - t_start - t_assembly
        
        if progress_callback:
            progress_callback({"message": "Calculating current density..."})
            
        # 電流密度の算出 j = -κ ∇φ
        j_e_z = np.zeros((nz, nx))
        j_e_x = np.zeros((nz, nx))
        
        for i in range(nz-1):
            j_e_z[i, :] = - 0.5 * (keff[i, :] + keff[i+1, :]) * (phi_e[i+1, :] - phi_e[i, :]) / dz
        for j in range(nx):
            j_next = (j + 1) % nx
            j_e_x[:, j] = - 0.5 * (keff[:, j] + keff[:, j_next]) * (phi_e[:, j_next] - phi_e[:, j]) / dx
            
        # 見かけの輸送抵抗 R = Δφ / I
        # セパレータ側(z=0)と集電体側(z=max)の平均電位差
        avg_phi_cc = np.mean(phi_e[-1, :])
        avg_phi_sep = np.mean(phi_e[0, :])
        apparent_resistance = abs(avg_phi_cc - avg_phi_sep) / self.config.applied_current_density
        
        res = ElectrolyteFieldResult(
            time=np.array([0.0]),
            phi_e=phi_e,
            j_e_x=j_e_x,
            j_e_z=j_e_z,
            z_coords_um=np.linspace(0, self.grid.thickness_um, nz),
            kpis={"apparent_resistance_ohm_m2": apparent_resistance},
            performance={
                "matrix_size": n_total,
                "assembly_time": t_assembly,
                "solve_time": t_solve
            }
        )
        return res

    def solve_transient_concentration(self, progress_callback=None) -> ElectrolyteFieldResult:
        """過渡濃度場を解く"""
        if self.prop_map is None:
            raise ValueError("Property map not prepared. Call prepare_property_map first.")
            
        import time
        t_start_total = time.time()
        
        nz, nx = self.grid.nz, self.grid.nx
        dz = self.grid.resolution * 1e-6
        dx = self.grid.resolution * 1e-6
        dt = self.config.dt
        t_final = self.config.t_final
        n_steps = int(t_final / dt)
        
        n_total = nz * nx
        
        if progress_callback:
            progress_callback({"message": f"Assembling concentration matrix ({nz}x{nx})..."})
            
        # 定数の準備
        deff = self.prop_map.deff_map
        eps = self.prop_map.porosity_map
        interface_mask = self.prop_map.interface_mask
        
        # ソース項の計算 (mol/s)
        # applied_current_density [A/m2] -> molar flux [mol/m2/s]
        # F = 96485 C/mol
        total_molar_flux = (self.config.applied_current_density / 96485.0) * (nx * dx)
        n_interface = np.sum(interface_mask)
        source_per_pixel = total_molar_flux / n_interface if n_interface > 0 else 0.0
        
        # 行列組み立て (左辺 A)
        # ε/dt * c_new - ∇·(Deff ∇c_new) = ε/dt * c_old + S
        A = sparse.lil_matrix((n_total, n_total))
        
        for i in range(nz):
            if progress_callback and i % 10 == 0:
                progress_callback({"message": f"Assembling matrix: row {i}/{nz}"})
            for j in range(nx):
                idx = i * nx + j
                
                # 境界条件: セパレータ側 (z=0)
                if i == 0 and self.config.bc_separator == "constant_concentration":
                    A[idx, idx] = 1.0
                    continue
                
                # 時間項
                A[idx, idx] += eps[i, j] / dt
                
                # 拡散項
                # Z方向
                if i < nz - 1:
                    d_half = 0.5 * (deff[i, j] + deff[i+1, j])
                    coeff = d_half / dz**2
                    A[idx, idx] += coeff
                    A[idx, (i+1)*nx + j] -= coeff
                
                if i > 0:
                    d_half_prev = 0.5 * (deff[i, j] + deff[i-1, j])
                    coeff_prev = d_half_prev / dz**2
                    A[idx, idx] += coeff_prev
                    A[idx, (i-1)*nx + j] -= coeff_prev
                elif self.config.bc_separator == "constant_flux":
                    # Flux BC at z=0 handled in b vector
                    pass
                
                # X方向
                j_next = (j + 1) % nx if self.config.bc_sides == "periodic" else min(j + 1, nx - 1)
                if j < nx - 1 or self.config.bc_sides == "periodic":
                    d_half_x = 0.5 * (deff[i, j] + deff[i, j_next])
                    coeff_x = d_half_x / dx**2
                    if not (self.config.bc_sides != "periodic" and j == nx - 1):
                        A[idx, idx] += coeff_x
                        A[idx, i*nx + j_next] -= coeff_x
                
                j_prev = (j - 1) % nx if self.config.bc_sides == "periodic" else max(j - 1, 0)
                if j > 0 or self.config.bc_sides == "periodic":
                    d_half_x_prev = 0.5 * (deff[i, j] + deff[i, j_prev])
                    coeff_x_prev = d_half_x_prev / dx**2
                    if not (self.config.bc_sides != "periodic" and j == 0):
                        A[idx, idx] += coeff_x_prev
                        A[idx, i*nx + j_prev] -= coeff_x_prev

        A_csr = A.tocsr()
        
        # 時間発展ループ
        c_current = np.full(n_total, self.config.c_initial)
        save_interval = self.config.save_interval_steps
        
        times = []
        c_history = []
        
        t_start_loop = time.time()
        for step in range(n_steps + 1):
            t = step * dt
            
            if step % save_interval == 0:
                times.append(t)
                c_history.append(c_current.reshape((nz, nx)).copy())
                
            if progress_callback and (step % 5 == 0 or step == n_steps):
                time_now = time.time()
                elapsed = time_now - t_start_loop
                eta = (elapsed / step * (n_steps - step)) if step > 0 else 0
                progress_callback({
                    "step": step,
                    "n_steps": n_steps,
                    "message": f"Time stepping: {t:.1f}s / {t_final:.1f}s (step {step}/{n_steps})",
                    "eta": eta
                })

            if step == n_steps:
                break
                
            # 右辺 b
            b = (eps.flatten() / dt) * c_current
            
            # 境界条件 z=0 (flux)
            if self.config.bc_separator == "constant_flux":
                # 入ってくる流束をソースとして加算 (z=0のピクセルに適用)
                # 簡易的な実装。本来は ghost cell 等を使うが、MVPなので。
                b[:nx] += (self.config.bc_value_separator / dz)
            elif self.config.bc_separator == "constant_concentration":
                # Dirichlet BC handled by fixing A and b
                b[:nx] = self.config.bc_value_separator
                
            # ソース項 (放電時はリチウムイオンを電解液に放出 -> c_e増加)
            # 実際の符号は discharge/charge に依存するが、ここでは applied_current に従う
            if n_interface > 0:
                b_source = np.zeros(n_total)
                b_source[interface_mask.flatten()] = source_per_pixel / (dx * dz)
                b += b_source
                
            # 解く
            c_current = spsolve(A_csr, b)
            
            # 負の濃度防止
            c_current = np.maximum(c_current, 1e-6)

        c_history_arr = np.array(c_history)
        times_arr = np.array(times)
        
        # 集約 (x方向平均)
        c_e_avg = np.mean(c_history_arr, axis=2)
        
        # KPI: 局所枯渇率 (濃度が初期の10%以下になった領域の割合)
        depletion_threshold = 0.1 * self.config.c_initial
        depleted_mask = (c_history_arr[-1] < depletion_threshold) & self.prop_map.pore_mask
        n_pore = np.sum(self.prop_map.pore_mask)
        depletion_rate = np.sum(depleted_mask) / n_pore if n_pore > 0 else 0.0
        
        res = ElectrolyteFieldResult(
            time=times_arr,
            c_e=c_history_arr,
            c_e_avg=c_e_avg,
            z_coords_um=np.linspace(0, self.grid.thickness_um, nz),
            kpis={
                "final_depletion_rate": depletion_rate,
                "min_c_e": np.min(c_history_arr[-1]),
                "avg_c_e_final": np.mean(c_history_arr[-1])
            },
            performance={
                "n_steps": n_steps,
                "total_solve_time": time.time() - t_start_total
            }
        )
        return res

def save_transport_results(output_dir: str, config: TransportSolverConfig, 
                           res_steady: Optional[ElectrolyteFieldResult],
                           res_transient: Optional[ElectrolyteFieldResult],
                           prop_map: PropertyMap):
    """輸送計算結果をファイルに保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Config
    import yaml
    with open(os.path.join(output_dir, "solver_config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, allow_unicode=True)
        
    # KPIs
    kpis = {}
    if res_steady:
        kpis.update(res_steady.kpis)
    if res_transient:
        kpis.update(res_transient.kpis)
        
    import pandas as pd
    pd.DataFrame([kpis]).to_csv(os.path.join(output_dir, "transport_kpis.csv"), index=False)
    
    # Fields (npz)
    save_dict = {}
    if res_steady:
        save_dict["phi_e_steady"] = res_steady.phi_e
        save_dict["j_e_x_steady"] = res_steady.j_e_x
        save_dict["j_e_z_steady"] = res_steady.j_e_z
    if res_transient:
        save_dict["times"] = res_transient.time
        save_dict["c_e_history"] = res_transient.c_e
        save_dict["c_e_avg"] = res_transient.c_e_avg
        save_dict["z_coords_um"] = res_transient.z_coords_um
    
    np.savez_compressed(os.path.join(output_dir, "transport_fields.npz"), **save_dict)
