import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage

@dataclass
class MicrostructureConfig:
    """2D構造生成の入力条件を保持する設定オブジェクト"""
    case_name: str = "default"
    width_um: float = 100.0
    thickness_um: float = 50.0
    resolution_um_per_px: float = 0.5
    phase_mode: str = "pore_active"  # "pore_active", "pore_active_cbd"
    target_porosity: float = 0.3
    target_active_fraction: float = 0.6
    target_cbd_fraction: float = 0.1
    particle_size_distribution_type: str = "monomodal"  # "monomodal", "bimodal", "custom"
    particle_size_distribution_parameters: Dict[str, Any] = field(default_factory=lambda: {"radius_um": 5.0})
    bimodal_ratio: float = 0.0  # 小粒子の比率
    thickness_gradient_parameters: Optional[Dict[str, Any]] = None
    in_plane_heterogeneity_parameters: Optional[Dict[str, Any]] = None
    calendaring_ratio: float = 1.0  # 1.0 = 圧縮なし
    random_seed: int = 42
    generation_method: str = "rsa"  # Random Sequential Addition
    postprocess_options: Dict[str, Any] = field(default_factory=dict)
    export_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict):
        # Dictからdataclassを作成する際に、不要なキーを除去するなどの処理を入れることも可能
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

class MicrostructureGrid:
    """
    2D断面の空間格子情報を保持するオブジェクト。
    x を面内方向、z を厚み方向。
    z=0 をセパレータ側、z=max を集電体側。
    """
    def __init__(self, config: MicrostructureConfig):
        self.width_um = config.width_um
        self.thickness_um = config.thickness_um
        self.resolution = config.resolution_um_per_px
        
        self.nx = int(np.round(self.width_um / self.resolution))
        self.nz = int(np.round(self.thickness_um / self.resolution))
        
        self.x_coords = np.arange(self.nx) * self.resolution
        self.z_coords = np.arange(self.nz) * self.resolution
        
    def um_to_px(self, val_um):
        return int(np.round(val_um / self.resolution))
    
    def px_to_um(self, val_px):
        return val_px * self.resolution

class PhaseMap:
    """
    2Dラベルマップを保持する。
    0=pore, 1=active material, 2=CBD
    """
    PORE = 0
    ACTIVE = 1
    CBD = 2
    
    def __init__(self, grid: MicrostructureGrid, data: Optional[np.ndarray] = None):
        self.grid = grid
        if data is not None:
            self.data = data
        else:
            self.data = np.zeros((grid.nz, grid.nx), dtype=np.int8)

    def get_fractions(self) -> Dict[str, float]:
        total_px = self.data.size
        fractions = {
            "pore": float(np.sum(self.data == self.PORE) / total_px),
            "active": float(np.sum(self.data == self.ACTIVE) / total_px),
            "cbd": float(np.sum(self.data == self.CBD) / total_px)
        }
        return fractions

    def get_mask(self, phase_label: int) -> np.ndarray:
        return self.data == phase_label

@dataclass
class TransportMap:
    """
    相マップから派生計算される輸送用マップを保持するオブジェクト。
    """
    pore_mask: np.ndarray
    active_mask: np.ndarray
    cbd_mask: np.ndarray
    local_porosity_map: Optional[np.ndarray] = None
    local_solid_fraction_map: Optional[np.ndarray] = None
    connectivity_map: Optional[np.ndarray] = None
    pore_path_availability: bool = False
    thickness_direction_path_existence: bool = False
    local_tortuosity_surrogate: Optional[np.ndarray] = None
    effective_property_placeholders: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Particle:
    """粒子個別のメタデータ"""
    particle_id: int
    center_x: float
    center_z: float
    radius_x: float
    radius_z: float
    phase_type: int = 1 # ACTIVE
    is_deformed: bool = False
    overlap_resolved: bool = True

class MicrostructureGenerator:
    """2D構造生成アルゴリズムを司るクラス"""
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.grid = MicrostructureGrid(config)
        self.particles: List[Particle] = []
        self.rng = np.random.default_rng(config.random_seed)

    def generate(self, progress_callback=None) -> PhaseMap:
        self.particles = []
        phase_map = PhaseMap(self.grid)
        
        target_area = self.grid.width_um * self.grid.thickness_um * self.config.target_active_fraction
        current_area = 0.0
        
        # 粒子生成
        max_total_attempts = 100000
        total_attempts = 0
        consecutive_failures = 0
        max_consecutive_failures = 2000
        
        while current_area < target_area and total_attempts < max_total_attempts:
            r = self._sample_radius()
            cx, cz = self._sample_position(r)
            
            if not self._check_overlap(cx, cz, r, r):
                p = Particle(
                    particle_id=len(self.particles),
                    center_x=cx,
                    center_z=cz,
                    radius_x=r,
                    radius_z=r
                )
                self.particles.append(p)
                current_area += np.pi * r * r
                consecutive_failures = 0
                
                if progress_callback and len(self.particles) % 10 == 0:
                    progress_callback({
                        "step": current_area,
                        "n_steps": target_area,
                        "message": f"Placing particles: {current_area/target_area*100:.1f}% (count: {len(self.particles)})"
                    })
            else:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    break
            
            total_attempts += 1
                
        if total_attempts >= max_total_attempts or consecutive_failures > max_consecutive_failures:
            msg = f"Warning: Generation stopped. Target reached: {current_area/target_area*100:.1f}%"
            print(msg)
            if progress_callback:
                progress_callback({"message": msg})

        # カレンダー圧縮変形
        if self.config.calendaring_ratio < 1.0:
            if progress_callback:
                progress_callback({"message": "Applying calendaring..."})
            self._apply_calendaring()

        # レンダリング
        if progress_callback:
            progress_callback({"message": "Rendering particles to grid..."})
        self._render_particles(phase_map)
        
        # CBD付与
        if self.config.phase_mode == "pore_active_cbd" or self.config.target_cbd_fraction > 0:
            if progress_callback:
                progress_callback({"message": "Adding CBD phase..."})
            self._add_cbd(phase_map)
            
        return phase_map

    def _sample_radius(self) -> float:
        params = self.config.particle_size_distribution_parameters
        dist_type = self.config.particle_size_distribution_type
        
        if dist_type == "monomodal":
            return float(params.get("radius_um", 5.0))
        
        elif dist_type == "bimodal":
            ratio_small = self.config.bimodal_ratio
            if self.rng.random() < ratio_small:
                return float(params.get("radius_small_um", 2.0))
            else:
                return float(params.get("radius_large_um", 8.0))
        
        elif dist_type == "custom":
            # 簡易実装: 平均と分散
            mean = params.get("mean_um", 5.0)
            std = params.get("std_um", 1.0)
            return float(max(0.1, self.rng.normal(mean, std)))
            
        return 5.0

    def _sample_position(self, r: float) -> Tuple[float, float]:
        # 基本は一様分布
        # 厚み方向勾配や面内不均一がある場合はここで重み付けサンプリング
        
        width = self.grid.width_um
        thickness = self.grid.thickness_um
        
        # 境界条件を考慮 (粒子がはみ出さないようにするか、周期境界にするか)
        # 今回ははみ出さないように制約を入れる
        cx = self.rng.uniform(r, width - r)
        
        # 厚み方向勾配の考慮
        if self.config.thickness_gradient_parameters:
            # z方向に偏りを持たせる (Rejection sampling or inversion transform)
            # 簡易的にセパレータ側(z=0)か集電体側(z=max)に寄せられるようにする
            cz = self._sample_z_with_gradient(r, thickness)
        else:
            cz = self.rng.uniform(r, thickness - r)
            
        # 面内不均一の考慮
        if self.config.in_plane_heterogeneity_parameters:
            cx = self._sample_x_with_heterogeneity(cx, r, width)
            
        return cx, cz

    def _sample_z_with_gradient(self, r: float, thickness: float) -> float:
        # 簡易的な線形勾配
        # "type": "linear", "slope": 0.5 (正なら集電体側に多い)
        params = self.config.thickness_gradient_parameters
        g_type = params.get("type", "linear")
        
        if g_type == "linear":
            slope = params.get("slope", 0.0) # -1 to 1
            # PDF(z) = 1 + slope * (2*z/L - 1)
            # CDF(z) = z/L + (slope/L) * (z^2/L - z)
            # 反転関数法は面倒なので Rejection sampling
            for _ in range(100):
                z = self.rng.uniform(r, thickness - r)
                pdf_val = 1.0 + slope * (2.0 * z / thickness - 1.0)
                if self.rng.random() < pdf_val / (1.0 + abs(slope)):
                    return z
        return self.rng.uniform(r, thickness - r)

    def _sample_x_with_heterogeneity(self, default_x: float, r: float, width: float) -> float:
        # 面内の粗密
        params = self.config.in_plane_heterogeneity_parameters
        # 簡易的に特定の位置に集まりやすくする
        return default_x # 未実装

    def _check_overlap(self, cx: float, cz: float, rx: float, rz: float) -> bool:
        for p in self.particles:
            # 楕円同士の厳密な判定は重いので、簡易的に距離判定
            # 円の場合は sqrt((dx)^2 + (dz)^2) < r1 + r2
            dx = cx - p.center_x
            dz = cz - p.center_z
            # 簡易的な楕円重なり判定 (等価半径)
            dist_sq = dx**2 + dz**2
            min_dist = (rx + p.radius_x + rz + p.radius_z) / 2.0
            if dist_sq < min_dist**2:
                return True
        return False

    def _apply_calendaring(self):
        ratio = self.config.calendaring_ratio
        # z方向に押し潰す
        for p in self.particles:
            p.radius_z *= ratio
            p.center_z *= ratio
            p.is_deformed = True
        
        # Gridサイズも変えるべきか？ 
        # config.thickness_um は圧縮前の厚みとし、grid.thickness_um を更新する
        old_thickness = self.grid.thickness_um
        new_thickness = old_thickness * ratio
        self.grid.thickness_um = new_thickness
        self.grid.nz = int(np.round(new_thickness / self.grid.resolution))
        self.grid.z_coords = np.arange(self.grid.nz) * self.grid.resolution

    def _render_particles(self, phase_map: PhaseMap):
        nz, nx = self.grid.nz, self.grid.nx
        res = self.grid.resolution
        
        # 高速化のため、全ピクセルに対してループ回すのではなく、粒子のBounding Box内のみ計算
        for p in self.particles:
            min_z_px = max(0, int((p.center_z - p.radius_z) / res))
            max_z_px = min(nz, int((p.center_z + p.radius_z) / res) + 1)
            min_x_px = max(0, int((p.center_x - p.radius_x) / res))
            max_x_px = min(nx, int((p.center_x + p.radius_x) / res) + 1)
            
            for iz in range(min_z_px, max_z_px):
                for ix in range(min_x_px, max_x_px):
                    z_um = iz * res
                    x_um = ix * res
                    if ((x_um - p.center_x) / p.radius_x)**2 + ((z_um - p.center_z) / p.radius_z)**2 <= 1.0:
                        phase_map.data[iz, ix] = PhaseMap.ACTIVE

    def _add_cbd(self, phase_map: PhaseMap):
        # 簡易実装: active粒子の周囲に付与、またはporeをランダムに埋める
        target_cbd = self.config.target_cbd_fraction
        if target_cbd <= 0:
            return
            
        # activeの周囲1pxをCBDにするなど
        # ここでは簡易的に、poreの一部をランダムにCBDに置き換える (連結性を壊しすぎない程度に)
        pore_mask = (phase_map.data == PhaseMap.PORE)
        pore_indices = np.argwhere(pore_mask)
        num_to_fill = int(target_cbd * phase_map.data.size)
        
        if len(pore_indices) > num_to_fill:
            fill_indices = self.rng.choice(len(pore_indices), num_to_fill, replace=False)
            for idx in fill_indices:
                iz, ix = pore_indices[idx]
                phase_map.data[iz, ix] = PhaseMap.CBD

class MicrostructureAnalyzer:
    """構造指標の算出を行うクラス"""
    def __init__(self, phase_map: PhaseMap):
        self.phase_map = phase_map
        self.grid = phase_map.grid

    def analyze(self, window_size_um: float = 5.0) -> TransportMap:
        pore_mask = self.phase_map.get_mask(PhaseMap.PORE)
        active_mask = self.phase_map.get_mask(PhaseMap.ACTIVE)
        cbd_mask = self.phase_map.get_mask(PhaseMap.CBD)
        
        # 1. 局所空隙率マップ (スライディングウィンドウ)
        local_porosity = self._calc_local_porosity(window_size_um)
        
        # 2. 連結性解析
        conn_info = self._analyze_connectivity(pore_mask)
        
        # 3. Tortuosity surrogate
        tortuosity = self._calc_tortuosity_surrogate(pore_mask)
        
        return TransportMap(
            pore_mask=pore_mask,
            active_mask=active_mask,
            cbd_mask=cbd_mask,
            local_porosity_map=local_porosity,
            connectivity_map=conn_info["labeled_map"],
            pore_path_availability=conn_info["path_available"],
            thickness_direction_path_existence=conn_info["path_available"],
            local_tortuosity_surrogate=tortuosity
        )

    def _calc_local_porosity(self, window_size_um: float) -> np.ndarray:
        window_px = max(1, self.grid.um_to_px(window_size_um))
        pore_float = self.phase_map.get_mask(PhaseMap.PORE).astype(float)
        return ndimage.uniform_filter(pore_float, size=window_px)

    def _analyze_connectivity(self, mask: np.ndarray) -> Dict[str, Any]:
        labeled_map, num_features = ndimage.label(mask)
        first_row_labels = set(labeled_map[0, :])
        last_row_labels = set(labeled_map[-1, :])
        first_row_labels.discard(0)
        last_row_labels.discard(0)
        common_labels = first_row_labels.intersection(last_row_labels)
        path_available = len(common_labels) > 0
        
        return {
            "labeled_map": labeled_map,
            "path_available": path_available,
            "num_features": num_features
        }

    def _calc_tortuosity_surrogate(self, mask: np.ndarray) -> np.ndarray:
        # 簡易的な厚み方向貫通率の指標 (1.0 = 直線)
        # Phase 2では定数値または局所空隙率の関数としての近似でもよい
        return np.ones_like(mask, dtype=float)

    def get_summary_statistics(self, transport_map: TransportMap, config: MicrostructureConfig) -> Dict[str, Any]:
        fractions = self.phase_map.get_fractions()
        
        # z方向の空隙率プロファイル
        z_porosity = np.mean(transport_map.pore_mask, axis=1)
        
        stats = {
            "target_porosity": config.target_porosity,
            "realized_porosity": fractions["pore"],
            "target_active_fraction": config.target_active_fraction,
            "realized_active_fraction": fractions["active"],
            "realized_cbd_fraction": fractions["cbd"],
            "connected_pore_fraction": float(np.sum(transport_map.connectivity_map > 0) / transport_map.pore_mask.size),
            "through_thickness_connectivity": transport_map.thickness_direction_path_existence,
            "mean_z_porosity": float(np.mean(z_porosity)),
            "std_z_porosity": float(np.std(z_porosity))
        }
        return stats

def visualize_microstructure(phase_map: PhaseMap, transport_map: TransportMap, save_path: Optional[str] = None):
    """構造生成結果の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Phase Map
    axes[0, 0].imshow(phase_map.data, origin='lower', cmap='viridis')
    axes[0, 0].set_title("Phase Map (0:Pore, 1:AM, 2:CBD)")
    
    # 2. Local Porosity
    im = axes[0, 1].imshow(transport_map.local_porosity_map, origin='lower', cmap='plasma')
    plt.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title("Local Porosity Map")
    
    # 3. Connectivity
    axes[1, 0].imshow(transport_map.connectivity_map, origin='lower', cmap='tab20')
    axes[1, 0].set_title("Connected Pore Components")
    
    # 4. Z-Porosity Profile
    z_porosity = np.mean(transport_map.pore_mask, axis=1)
    axes[1, 1].plot(z_porosity, np.arange(len(z_porosity)) * phase_map.grid.resolution)
    axes[1, 1].set_xlabel("Porosity")
    axes[1, 1].set_ylabel("Z [um] (Separator -> Collector)")
    axes[1, 1].set_title("Through-thickness Porosity Profile")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def save_microstructure_results(output_dir: str, config: MicrostructureConfig, phase_map: PhaseMap, transport_map: TransportMap, stats: Dict[str, Any], particles: List[Particle]):
    """結果の保存"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Config
    with open(out_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, allow_unicode=True)
    
    # 2. Phase Map (Array)
    np.savez_compressed(out_path / "phase_map.npz", data=phase_map.data)
    
    # 3. Stats (CSV)
    import csv
    with open(out_path / "stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
        
    # 4. Particles (JSON)
    with open(out_path / "particles.json", "w", encoding="utf-8") as f:
        # Particleオブジェクトをdictに変換して保存
        json_data = []
        for p in particles:
            p_dict = {k: v for k, v in p.__dict__.items()}
            json_data.append(p_dict)
        json.dump(json_data, f, indent=2)
        
    # 5. Preview Image
    visualize_microstructure(phase_map, transport_map, save_path=str(out_path / "preview.png"))
    plt.close()
