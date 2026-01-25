from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto

class TaskType(Enum):
    SMALL_EXP = auto()
    MID_EXP = auto()
    FIN_EXP = auto()
    DR_REVIEW = auto()

@dataclass
class Approver:
    approver_id: str
    approver_type: str # 'Senior', 'Coordinator', 'New'
    capacity: int
    quality: float

# Step 6: 承認者タイプの定義
APPROVER_TYPES = {
    "Senior": {"capacity": 7, "quality": 0.76},
    "Coordinator": {"capacity": 3, "quality": 0.7},
    "New": {"capacity": 1, "quality": 0.4}
}

@dataclass
class Task:
    task_id: str
    task_type: TaskType
    duration_days: float
    generated_by: Optional[str] = None # 'REWORK' or None
    created_at: float = 0.0

@dataclass
class Job:
    job_id: str
    created_at: float
    current_node: Optional[str] = None
    bundle_items: List['Job'] = field(default_factory=list)
    rework_weight: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    rework_count: int = 0
    temp_enqueue_time: float = 0.0 # 待ち時間計算用の一時変数
    
    # Ver13 added: AnyLogic features
    latent: Optional[Any] = None # LatentRisk
    work_packages: List[Any] = field(default_factory=list) # List[WorkPackage]
    
    def add_history(self, node_id: str, event: str, time: float, **kwargs):
        entry = {
            "node_id": node_id,
            "event": event,
            "time": time
        }
        entry.update(kwargs)
        self.history.append(entry)
