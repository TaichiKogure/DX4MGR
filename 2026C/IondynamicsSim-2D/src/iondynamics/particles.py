import numpy as np
from typing import Tuple, List
from iondynamics.config import ParticleConfig

def generate_particles(mode: str, n: int, bbox_um: List[float], radius_spec: float, seed: int = 42, bimodal_spec=None) -> np.ndarray:
    """
    戻り値: ndarray shape=(n, 4) -> [x, y, z, radius] (単位: um)
    """
    np.random.seed(seed)
    Lx, Ly, Lz = bbox_um
    
    # 粒子データの初期化 [x, y, z, r]
    particles = np.zeros((n, 4))
    
    if mode == "random":
        particles[:, 0] = np.random.uniform(0, Lx, n)
        particles[:, 1] = np.random.uniform(0, Ly, n)
        if Lz > 0:
            particles[:, 2] = np.random.uniform(0, Lz, n)
        particles[:, 3] = radius_spec
        
    elif mode == "regular":
        # 簡易的な格子状配置
        # n個を立方格子状に並べるための分割数を計算
        if Lz > 0:
            nx = int(np.power(n * Lx*Lx / (Ly*Lz), 1/3)) # 簡易計算
            # 実際にはもっと真面目にやる必要があるが、MVPなので簡易的に
            nx = int(np.ceil(np.power(n, 1/3)))
            ny = int(np.ceil(np.power(n, 1/3)))
            nz = int(np.ceil(np.power(n, 1/3)))
        else:
            nx = int(np.ceil(np.sqrt(n * Lx / Ly)))
            ny = int(np.ceil(n / nx))
            nz = 1
        
        xs = np.linspace(0, Lx, nx)
        ys = np.linspace(0, Ly, ny)
        zs = [0] if Lz == 0 else np.linspace(0, Lz, nz)
        
        idx = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    if idx < n:
                        particles[idx] = [x, y, z, radius_spec]
                        idx += 1
                        
    elif mode == "bimodal":
        if bimodal_spec is None:
            raise ValueError("bimodal_spec is required for bimodal mode")
        
        n_large = int(n * bimodal_spec.ratio_large)
        n_small = n - n_large
        
        # 大粒子
        particles[:n_large, 0] = np.random.uniform(0, Lx, n_large)
        particles[:n_large, 1] = np.random.uniform(0, Ly, n_large)
        if Lz > 0:
            particles[:n_large, 2] = np.random.uniform(0, Lz, n_large)
        particles[:n_large, 3] = bimodal_spec.radius_large_um
        
        # 小粒子
        particles[n_large:, 0] = np.random.uniform(0, Lx, n_small)
        particles[n_large:, 1] = np.random.uniform(0, Ly, n_small)
        if Lz > 0:
            particles[n_large:, 2] = np.random.uniform(0, Lz, n_small)
        particles[n_large:, 3] = bimodal_spec.radius_small_um
        
    return particles

def map_concentration_to_particles(particles: np.ndarray, x_profile: np.ndarray, c_profile: np.ndarray, thickness_um: float):
    """
    各粒子のx座標に応じて濃度値を補間割り当てする。
    particles: [n, 4] (um)
    x_profile: pybammの距離 (m)
    c_profile: pybammの濃度 (mol/m3), shape=(Nx, Nt)
    thickness_um: 電極厚み
    戻り値: shape=(n, Nt) の時系列濃度データ
    """
    # 粒子のx座標 (um) を m に変換
    # PyBaMMのxは0からLまでの範囲
    # 粒子のxは厚み方向 (default.yamlでは 200, 80, 0 なので 80 が厚みか？)
    # bbox_um: [幅, 厚み, 奥行き] なので index 1 が厚み方向 (x)
    p_x_um = particles[:, 1] # 厚み方向をyとしている場合は1
    # 指示書では bbox は (Lx, Ly, Lz) で電極幅・厚み・奥行き。
    # つまり Lx=幅, Ly=厚み, Lz=奥行き。 厚み方向は index 1 (y)。
    
    p_x_m = p_x_um * 1e-6
    
    # x_profile (m) を使って補間
    # c_profile は (Nx, Nt)
    n_particles = particles.shape[0]
    n_time = c_profile.shape[1]
    
    particle_c_series = np.zeros((n_particles, n_time))
    
    for t in range(n_time):
        particle_c_series[:, t] = np.interp(p_x_m, x_profile, c_profile[:, t])
        
    return particle_c_series
