import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import imageio_ffmpeg
import matplotlib
from iondynamics.simulate import SimResult

# ffmpegのパスを自動設定
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

def animate_particles(result: SimResult, particles: np.ndarray, particle_c: np.ndarray, out_path: str):
    """
    particles: (n, 4) -> [x, y, z, r] (um)
    particle_c: (n, Nt) -> concentration
    """
    # 出力ディレクトリの作成
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左パネル: 粒子
    # 2D表示。bboxのxを横、yを縦（厚み）とする。
    # map_concentration_to_particles では y(index 1) を厚みとしている
    scatter = ax1.scatter(particles[:, 0], particles[:, 1], 
                          s=particles[:, 3]**2, # 面積に比例
                          c=particle_c[:, 0], cmap='RdYlBu_r', edgecolors='k', alpha=0.7)
    
    ax1.set_xlim(0, result.config.particles.bbox_um[0])
    ax1.set_ylim(0, result.config.particles.bbox_um[1])
    ax1.set_aspect('equal')
    ax1.set_xlabel("Width [um]")
    ax1.set_ylabel("Thickness [um]")
    
    # 右パネル: 電圧
    line, = ax2.plot(result.time, result.voltage)
    point, = ax2.plot([result.time[0]], [result.voltage[0]], 'ro')
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Voltage [V]")
    
    # カラーバー
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Concentration [mol/m3]")
    
    def update(frame):
        # 粒子の色更新
        scatter.set_array(particle_c[:, frame])
        
        # 電圧プロットの現在地更新
        point.set_data([result.time[frame]], [result.voltage[frame]])
        
        # タイトル更新
        soc = 100 * (1 - frame / len(result.time)) # 簡易的なSOC
        ax1.set_title(f"t = {result.time[frame]:.1f} s, SOC ~ {soc:.1f}%, C-rate = {result.config.operation.c_rate}")
        return scatter, point

    fps = result.config.output.fps
    # 描画が多すぎると重いので適宜間引く
    step = max(1, len(result.time) // (fps * 10)) # 約10秒の動画
    frames = range(0, len(result.time), step)
    
    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    
    if out_path.endswith('.mp4'):
        ani.save(out_path, writer='ffmpeg', fps=fps)
    else:
        ani.save(out_path, writer='pillow', fps=fps)
    
    plt.close(fig)
