from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Callable
from core.entities import Job, Task

class GateNode(ABC):
    def __init__(self, node_id: str, engine: Any):
        self.node_id = node_id
        self.engine = engine
        self.queue: List[Job] = []
        self.processed_count = 0
        self.total_wait_time = 0.0

    @abstractmethod
    def enqueue(self, job: Job, now: float):
        """Jobをキューに追加し、enqueue_timeを記録する"""
        pass

    @abstractmethod
    def can_process(self, now: float) -> bool:
        """処理開始可能かどうかを返す"""
        pass

    @abstractmethod
    def process(self, now: float):
        """Jobを処理し、次の遷移をスケジュールする"""
        pass

    def stats(self) -> dict:
        """統計情報を返す"""
        return {
            "node_id": self.node_id,
            "queue_length": len(self.queue),
            "processed_count": self.processed_count,
            "avg_wait_time": self.total_wait_time / self.processed_count if self.processed_count > 0 else 0
        }

class WorkGate(GateNode):
    def __init__(self, node_id: str, engine: Any, n_servers: int, duration_dist: Callable, next_node_id: Optional[str]):
        super().__init__(node_id, engine)
        self.n_servers = n_servers
        self.busy_servers = 0
        self.duration_dist = duration_dist
        self.next_node_id = next_node_id

    def enqueue(self, job: Job, now: float):
        job.add_history(self.node_id, "ENQUEUE", now)
        job.temp_enqueue_time = now
        self.queue.append(job)

    def can_process(self, now: float) -> bool:
        return self.busy_servers < self.n_servers and len(self.queue) > 0

    def process(self, now: float):
        if not self.can_process(now):
            return
        
        job = self.queue.pop(0)
        self.busy_servers += 1
        
        wait_time = now - job.temp_enqueue_time
        self.total_wait_time += wait_time
        job.add_history(self.node_id, "START_WORK", now, wait_time=wait_time)
        
        duration = self.duration_dist()
        finish_time = now + duration
        
        # 完了イベントをスケジュール
        self.engine.schedule_event(finish_time, "WORK_COMPLETE", {"node_id": self.node_id, "job": job})

    def on_work_complete(self, job: Job, now: float):
        self.busy_servers -= 1
        self.processed_count += 1
        job.add_history(self.node_id, "COMPLETE_WORK", now)
        
        # 次のノードへ（ARRIVALイベントをスケジュール）
        self.engine.schedule_event(now, "ARRIVAL", {"job": job, "target_node": self.next_node_id}, priority=5)
        # 空いたサーバーで次のジョブを処理
        self.engine.check_node_activation(self.node_id)

class BundleGate(GateNode):
    def __init__(self, node_id: str, engine: Any, bundle_size_dist: Callable, next_node_id: str):
        super().__init__(node_id, engine)
        self.bundle_size_dist = bundle_size_dist
        self.next_node_id = next_node_id
        self.current_bundle_size = self.bundle_size_dist()

    def enqueue(self, job: Job, now: float):
        job.add_history(self.node_id, "ENQUEUE", now)
        job.temp_enqueue_time = now
        self.queue.append(job)

    def can_process(self, now: float) -> bool:
        return len(self.queue) >= self.current_bundle_size

    def process(self, now: float):
        if not self.can_process(now):
            return
        
        items = []
        for _ in range(self.current_bundle_size):
            item = self.queue.pop(0)
            wait_time = now - item.temp_enqueue_time
            self.total_wait_time += wait_time
            item.add_history(self.node_id, "BUNDLED", now, wait_time=wait_time)
            items.append(item)
        
        self.processed_count += 1
        
        # 新しいJob（バンドル）を作成
        bundle_job = Job(
            job_id=f"bundle_{self.node_id}_{self.processed_count}",
            created_at=now,
            bundle_items=items
        )
        bundle_job.add_history(self.node_id, "CREATED_BUNDLE", now)
        
        # 次のノードへ
        self.engine.schedule_event(now, "ARRIVAL", {"job": bundle_job, "target_node": self.next_node_id}, priority=5)
        
        # 次のバンドルサイズを決定
        self.current_bundle_size = self.bundle_size_dist()
        self.engine.check_node_activation(self.node_id)

class MeetingGate(GateNode):
    def __init__(self, node_id: str, engine: Any, period_days: float, approvers: List[Any], 
                 next_node_id: Optional[str], rework_node_id: Optional[str], 
                 rework_policy: Any, nogo_node_id: Optional[str] = None):
        super().__init__(node_id, engine)
        self.period_days = period_days
        self.approvers = approvers
        self.next_node_id = next_node_id
        self.rework_node_id = rework_node_id
        self.nogo_node_id = nogo_node_id
        self.rework_policy = rework_policy
        self.next_meeting_time = period_days 
        
        # 実効キャパシティと品質の計算 (Step 6)
        if approvers:
            self.capacity = sum(a.capacity for a in approvers)
            self.quality = sum(a.quality * a.capacity for a in approvers) / self.capacity if self.capacity > 0 else 0
        else:
            self.capacity = 0
            self.quality = 0
            
        # 会議周期イベントをスケジュール
        self.engine.schedule_event(self.next_meeting_time, "MEETING_START", {"node_id": self.node_id})

    def enqueue(self, job: Job, now: float):
        job.add_history(self.node_id, "ENQUEUE", now)
        job.temp_enqueue_time = now
        self.queue.append(job)

    def can_process(self, now: float) -> bool:
        return False

    def process(self, now: float):
        count = 0
        while self.queue and count < self.capacity:
            job = self.queue.pop(0)
            count += 1
            wait_time = now - job.temp_enqueue_time
            self.total_wait_time += wait_time
            self.processed_count += 1
            
            job.add_history(self.node_id, "REVIEW", now, wait_time=wait_time)
            
            # 判定 (Step 4.3 & Step 5)
            rand = self.engine.rng.random()
            if rand < self.quality: # GO
                self.engine.schedule_event(now, "ARRIVAL", {"job": job, "target_node": self.next_node_id}, priority=5)
            elif rand < self.quality + (1.0 - self.quality) * 0.8: # CONDITIONAL (差し戻し：適当に8割)
                job.rework_count += 1
                # 増殖ルールの適用 (Step 5)
                n_new = self.rework_policy.apply_rework(job, now)
                job.add_history(self.node_id, "REWORK_PROLIFERATED", now, n_new_tasks=n_new)
                self.engine.schedule_event(now, "ARRIVAL", {"job": job, "target_node": self.rework_node_id}, priority=5)
            else: # NO_GO
                self.engine.schedule_event(now, "ARRIVAL", {"job": job, "target_node": self.nogo_node_id}, priority=5)

        # 次の会議をスケジュール
        self.next_meeting_time += self.period_days
        self.engine.schedule_event(self.next_meeting_time, "MEETING_START", {"node_id": self.node_id})
