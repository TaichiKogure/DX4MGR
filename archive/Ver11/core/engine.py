import heapq
from typing import List, Dict, Any, Callable
from core.entities import Job
from core.gates import GateNode

class SimulationEngine:
    def __init__(self, rng=None, sampling_interval: float = 1.0):
        self.events = [] # (time, priority, counter, event_type, data)
        self.event_counter = 0
        self.now = 0.0
        self.nodes: Dict[str, GateNode] = {}
        self.rng = rng if rng else None # np.random.default_rng()
        self.sampling_interval = float(sampling_interval) if sampling_interval else 0.0
        self.next_sample_time = 0.0

        self.results = {
            "completed_jobs": [],
            "wip_history": []
        }

    def add_node(self, node: GateNode):
        self.nodes[node.node_id] = node

    def schedule_event(self, time: float, event_type: str, data: Any, priority: int = 10):
        heapq.heappush(self.events, (time, priority, self.event_counter, event_type, data))
        self.event_counter += 1

    def _snapshot_wip(self, at_time: float):
        node_wip = {}
        for node_id, node in self.nodes.items():
            in_queue = len(getattr(node, "queue", []))
            in_service = 0
            if hasattr(node, "busy_servers"):
                in_service = int(getattr(node, "busy_servers", 0))
            node_wip[node_id] = in_queue + in_service

        self.results["wip_history"].append({
            "time": float(at_time),
            "node_wip": node_wip,
            "total_wip": int(sum(node_wip.values()))
        })

    def run(self, max_days: float):
        # 最初のスナップショット
        if self.sampling_interval > 0:
            self.next_sample_time = 0.0
            self._snapshot_wip(0.0)

        while self.events:
            time, priority, counter, event_type, data = heapq.heappop(self.events)
            if time > max_days:
                self.now = max_days
                break

            self.now = time
            self.handle_event(event_type, data)

            # WIPサンプリング（一定間隔で）
            if self.sampling_interval > 0:
                while self.next_sample_time + self.sampling_interval <= self.now + 1e-12:
                    self.next_sample_time += self.sampling_interval
                    if self.next_sample_time <= max_days + 1e-12:
                        self._snapshot_wip(self.next_sample_time)

    def handle_event(self, event_type: str, data: Any):
        if event_type == "ARRIVAL":
            job = data["job"]
            target_node_id = data["target_node"]
            if target_node_id in self.nodes:
                self.nodes[target_node_id].enqueue(job, self.now)
                self.check_node_activation(target_node_id)
            else:
                # 終端ノード
                self.results["completed_jobs"].append(job)
            
        elif event_type == "PROCESS_READY":
            node_id = data["node_id"]
            if node_id in self.nodes:
                self.nodes[node_id].process(self.now)

        elif event_type == "WORK_COMPLETE":
            node_id = data["node_id"]
            job = data["job"]
            self.nodes[node_id].on_work_complete(job, self.now)

        elif event_type == "MEETING_START":
            node_id = data["node_id"]
            self.nodes[node_id].process(self.now)

    def check_node_activation(self, node_id: str):
        if self.nodes[node_id].can_process(self.now):
            # 即座に処理開始可能な場合はイベントをスケジュール
            self.schedule_event(self.now, "PROCESS_READY", {"node_id": node_id}, priority=8)

    def get_total_wip(self) -> int:
        total = 0
        for node in self.nodes.values():
            total += len(node.queue)
        return total
