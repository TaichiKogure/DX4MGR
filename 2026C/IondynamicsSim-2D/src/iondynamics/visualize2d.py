import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
from iondynamics.solver2d import ElectrolyteFieldResult, PropertyMap

def plot_2d_field(field: np.ndarray, title: str, cmap: str = "viridis", 
                  extent: Optional[list] = None, save_path: Optional[str] = None,
                  label: str = ""):
    """2Dフィールドのヒートマップを描画"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(field, cmap=cmap, extent=extent, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Width [um]")
    ax.set_ylabel("Thickness [um] (0=separator)")
    fig.colorbar(im, label=label)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

def plot_concentration_profiles(res: ElectrolyteFieldResult, save_path: Optional[str] = None):
    """厚み方向平均濃度の時間推移を描画"""
    fig, ax = plt.subplots(figsize=(8, 6))
    nt = len(res.time)
    indices = np.linspace(0, nt - 1, 6, dtype=int)
    
    for idx in indices:
        ax.plot(res.z_coords_um, res.c_e_avg[idx, :], label=f"t={res.time[idx]:.1f}s")
        
    ax.set_xlabel("Thickness [um] (0=separator)")
    ax.set_ylabel("Avg. Electrolyte Conc. [mol/m3]")
    ax.set_title("Thickness-direction Concentration Profile")
    ax.legend()
    ax.grid(True)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

def visualize_transport_results(res_steady: Optional[ElectrolyteFieldResult], 
                                res_transient: Optional[ElectrolyteFieldResult],
                                prop_map: PropertyMap,
                                grid_info: dict,
                                output_dir: str,
                                phase_map: Optional[any] = None):
    """輸送計算結果を一括可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    width = grid_info["width_um"]
    thickness = grid_info["thickness_um"]
    extent = [0, width, 0, thickness]
    
    # 0. 構造情報の保存
    if phase_map:
        # 相マップの可視化 (単純なimshow)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(phase_map.data, extent=extent, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Phase Map used for Simulation")
        ax.set_xlabel("Width [um]")
        ax.set_ylabel("Thickness [um]")
        fig.colorbar(im, label="Phase Label (0:pore, 1:AM, 2:CBD)")
        fig.savefig(os.path.join(output_dir, "phase_map_used.png"), bbox_inches="tight")
        plt.close(fig)
        
    # 界面マスクの可視化
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(prop_map.interface_mask, extent=extent, origin="lower", aspect="auto", cmap="gray_r")
    ax.set_title("Interface Mask (Reactive areas)")
    ax.set_xlabel("Width [um]")
    ax.set_ylabel("Thickness [um]")
    fig.savefig(os.path.join(output_dir, "interface_mask.png"), bbox_inches="tight")
    plt.close(fig)

    # 1. 物性マップ
    plot_2d_field(prop_map.deff_map, "Effective Diffusivity [m2/s]", 
                  extent=extent, save_path=os.path.join(output_dir, "deff_map.png"))
    plot_2d_field(prop_map.keff_map, "Effective Conductivity [S/m]", 
                  extent=extent, save_path=os.path.join(output_dir, "keff_map.png"))
    
    # 2. 定常ポテンシャル
    if res_steady and res_steady.phi_e is not None:
        plot_2d_field(res_steady.phi_e, "Electrolyte Potential [V]", 
                      extent=extent, save_path=os.path.join(output_dir, "phi_e_steady.png"),
                      label="Potential [V]")
        
        # 重ね描き
        fig, ax = plot_2d_field(res_steady.phi_e, "Potential with Microstructure", extent=extent, label="Potential [V]")
        # active material の輪郭を重ねる
        if phase_map is not None:
            ax.contour(phase_map.data, levels=[0.5], colors='white', extent=extent, linewidths=0.5)
        fig.savefig(os.path.join(output_dir, "phi_e_steady_with_microstructure.png"), bbox_inches="tight")
        plt.close(fig)

        # 電流密度 (ノルム)
        j_norm = np.sqrt(res_steady.j_e_x**2 + res_steady.j_e_z**2)
        plot_2d_field(j_norm, "Current Density Norm [A/m2]", 
                      extent=extent, save_path=os.path.join(output_dir, "j_norm_steady.png"),
                      label="Current Density [A/m2]")

    # 3. 過渡濃度
    if res_transient and res_transient.c_e is not None:
        # 最終時刻の濃度分布
        plot_2d_field(res_transient.c_e[-1], "Final Electrolyte Concentration [mol/m3]", 
                      extent=extent, save_path=os.path.join(output_dir, "ce_final_2d.png"),
                      label="Conc [mol/m3]")
        
        # 重ね描き
        fig, ax = plot_2d_field(res_transient.c_e[-1], "Conc with Microstructure", extent=extent, label="Conc [mol/m3]")
        if phase_map is not None:
            ax.contour(phase_map.data, levels=[0.5], colors='white', extent=extent, linewidths=0.5)
        fig.savefig(os.path.join(output_dir, "ce_final_2d_with_microstructure.png"), bbox_inches="tight")
        plt.close(fig)

        # 濃度プロファイル
        plot_concentration_profiles(res_transient, save_path=os.path.join(output_dir, "ce_profiles_1d.png"))
