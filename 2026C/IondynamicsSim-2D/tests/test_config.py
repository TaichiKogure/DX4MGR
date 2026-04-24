import pytest
import os
from iondynamics.config import load_config, SimConfig

def test_load_default_config():
    config_path = "configs/default.yaml"
    cfg = load_config(config_path)
    assert isinstance(cfg, SimConfig)
    assert cfg.meta.case_name == "default"
    assert cfg.electrode.thickness_um == 80
    assert cfg.particles.count == 300

def test_validate_config_fail():
    from iondynamics.config import validate_config, SimConfig, MetaConfig, ElectrodeConfig, ResistanceConfig, OperationConfig, ParticleConfig, OutputConfig
    
    # Invalid porosity
    bad_electrode = ElectrodeConfig(thickness_um=80, porosity=1.5, particle_radius_um=5.0)
    cfg = SimConfig(
        meta=MetaConfig("test", "test"),
        electrode=bad_electrode,
        resistances=ResistanceConfig(100.0, 1.0, 1e-14),
        operation=OperationConfig("discharge", 1.0, 2.5, 1.0),
        particles=ParticleConfig("random", 300, [200, 80, 0], 42),
        output=OutputConfig("mp4", 20, True)
    )
    
    with pytest.raises(ValueError, match="porosity"):
        validate_config(cfg)
