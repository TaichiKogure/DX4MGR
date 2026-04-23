from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path

@dataclass
class MetaConfig:
    case_name: str
    description: str

@dataclass
class ElectrodeConfig:
    thickness_um: float
    porosity: float
    particle_radius_um: float

@dataclass
class ResistanceConfig:
    electronic_conductivity_S_m: float
    ionic_conductivity_S_m: float
    solid_diffusivity_m2_s: float

@dataclass
class OperationConfig:
    mode: str
    c_rate: float
    cutoff_voltage_V: float
    initial_soc: float

@dataclass
class BimodalConfig:
    ratio_large: float
    radius_small_um: float
    radius_large_um: float

@dataclass
class ParticleConfig:
    mode: str
    count: int
    bbox_um: List[float]
    seed: int
    bimodal: Optional[BimodalConfig] = None

@dataclass
class OutputConfig:
    animation_format: str
    fps: int
    save_csv: bool

@dataclass
class SimConfig:
    meta: MetaConfig
    electrode: ElectrodeConfig
    resistances: ResistanceConfig
    operation: OperationConfig
    particles: ParticleConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, data: dict):
        # 手動でマッピング（小規模なのでこれで十分）
        return cls(
            meta=MetaConfig(**data['meta']),
            electrode=ElectrodeConfig(**data['electrode']),
            resistances=ResistanceConfig(**data['resistances']),
            operation=OperationConfig(**data['operation']),
            particles=ParticleConfig(
                mode=data['particles']['mode'],
                count=data['particles']['count'],
                bbox_um=data['particles']['bbox_um'],
                seed=data['particles']['seed'],
                bimodal=BimodalConfig(**data['particles']['bimodal']) if 'bimodal' in data['particles'] and data['particles']['bimodal'] else None
            ),
            output=OutputConfig(**data['output'])
        )

def load_config(path: str) -> SimConfig:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    config = SimConfig.from_dict(data)
    validate_config(config)
    return config

def validate_config(cfg: SimConfig):
    if cfg.electrode.thickness_um <= 0:
        raise ValueError("thickness_um must be positive")
    if not (0 < cfg.electrode.porosity < 1):
        raise ValueError("porosity must be between 0 and 1")
    if cfg.operation.c_rate <= 0:
        raise ValueError("c_rate must be positive")
    if cfg.operation.mode not in ["charge", "discharge"]:
        raise ValueError("mode must be 'charge' or 'discharge'")
    if cfg.particles.mode not in ["random", "regular", "bimodal"]:
        raise ValueError("particle mode must be 'random', 'regular', or 'bimodal'")
    
    # Warning for unusual values as per guidelines
    import logging
    logger = logging.getLogger(__name__)
    if cfg.electrode.porosity < 0.1:
        logger.warning(f"Porosity {cfg.electrode.porosity} is very low (< 0.1)")
    if cfg.electrode.thickness_um > 200:
        logger.warning(f"Electrode thickness {cfg.electrode.thickness_um} um is very high (> 200 um)")
