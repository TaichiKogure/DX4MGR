import pytest
import numpy as np
from iondynamics.particles import generate_particles, map_concentration_to_particles

def test_generate_particles_random():
    n = 100
    bbox = [200.0, 80.0, 0.0]
    radius = 5.0
    particles = generate_particles("random", n, bbox, radius)
    
    assert particles.shape == (n, 4)
    assert np.all(particles[:, 0] >= 0) and np.all(particles[:, 0] <= 200)
    assert np.all(particles[:, 1] >= 0) and np.all(particles[:, 1] <= 80)
    assert np.all(particles[:, 2] == 0)
    assert np.all(particles[:, 3] == radius)

def test_generate_particles_regular():
    n = 100
    bbox = [100.0, 100.0, 0.0]
    radius = 5.0
    particles = generate_particles("regular", n, bbox, radius)
    assert particles.shape[0] >= n # 格子状なのでぴったりにならないこともあるが今回は切り上げている

def test_map_concentration():
    n = 10
    particles = np.zeros((n, 4))
    particles[:, 1] = np.linspace(0, 80, n) # 厚み方向 0-80 um
    
    x_profile = np.array([0, 40e-6, 80e-6])
    c_profile = np.array([
        [1000, 900],
        [1100, 1000],
        [1200, 1100]
    ]) # (3, 2) Nx=3, Nt=2
    
    res = map_concentration_to_particles(particles, x_profile, c_profile, 80.0)
    assert res.shape == (n, 2)
    assert res[0, 0] == 1000
    assert res[-1, 0] == 1200
