from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from enum import Enum, auto

class TaskType(Enum):
    SMALL_EXP = auto()
    MID_EXP = auto()
    FIN_EXP = auto()
    DR_REVIEW = auto()
    PROTOTYPE = auto()
    TEST = auto()
    MASS_PROD = auto()

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
    "New": {"capacity": 1, "quality": 0.4},
    "Director": {"capacity": 2, "quality": 0.9}
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
    is_rework_task: bool = False
    parent_job_id: Optional[str] = None
    rework_source_gate: Optional[str] = None
    rework_task_type: Optional[TaskType] = None
    is_rejected: bool = False
    
    # Plan 6 統合: 技術要素への紐付け
    target_tech: Any = None 
    
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


# ===== Ver2: Cross-departmental entities (minimal implementation) =====

@dataclass
class Department:
    dept_id: str
    name: str
    calendar: Optional[Any] = None  # 将来的に稼働日/時間の表現（未使用: スタブ）
    cost_factor: float = 1.0
    sla: Dict[str, float] = field(default_factory=lambda: {"avg_response": 0.0, "max_wait": 0.0})


@dataclass
class Handoff:
    """部署間ハンドオフの最小表現。
    - q_if: インターフェース品質（0..1, 1が最高）
    - info_loss_lambda: 情報損失率λ（0..1の係数解釈）
    - transfer_time_dist: 転送/待機時間分布の設定（{'type': 'exponential', 'params': {'scale': 2}} 等）
    """
    from_dept: str
    to_dept: str
    q_if: float = 1.0
    info_loss_lambda: float = 0.0
    transfer_time_dist: Optional[Dict[str, Any]] = None

    def sampler(self, rng) -> Callable[[], float]:
        cfg = self.transfer_time_dist or {"type": "constant", "params": {"value": 0.0}}
        t = (cfg or {}).get("type", "constant")
        p = (cfg or {}).get("params", {}) or {}
        if t == "exponential":
            scale = float(p.get("scale", 1.0))
            return lambda: float(rng.exponential(scale))
        if t == "uniform":
            a = float(p.get("low", 0.0)); b = float(p.get("high", max(a, 1.0)))
            return lambda: float(rng.uniform(a, b))
        if t == "triangular":
            left = float(p.get("left", 0.0)); mode = float(p.get("mode", 0.5)); right = float(p.get("right", 1.0))
            return lambda: float(rng.triangular(left, mode, right))
        # constant
        val = float(p.get("value", 0.0))
        return lambda: val


class DecisionLogic(Enum):
    GO = auto()
    COND = auto()
    NO_GO = auto()


@dataclass
class CrossDeptMeeting:
    """部署横断会議の最小スタブ。
    - departments: 参加部署IDの集合
    - interval_days: 開催間隔（平均）
    - threshold: 判定用のしきい値（意味付けは簡易: WIPや遅延の代理指標）
    - logic: GO/COND/NO_GO の優先ロジック（簡易）
    """
    departments: List[str]
    interval_days: float = 14.0
    threshold: float = 0.0
    logic: DecisionLogic = DecisionLogic.GO


@dataclass
class WorkItem:
    """部門フローの宣言的仕様（最小）。
    - steps: [dept_id or {"AND": [..]} or {"OR": [..]}] の簡易表現
    - owners: 各工程の責任部署（任意）
    - rework_rules: 差し戻し規則（簡易: prob, max_cycles）
    注意: 現行のDESフローに直結はしない（Ver2最小実装: 将来の拡張用に保持）。
    """
    work_id: str
    steps: List[Any]
    owners: Optional[List[str]] = None
    rework_rules: Optional[Dict[str, Any]] = None
