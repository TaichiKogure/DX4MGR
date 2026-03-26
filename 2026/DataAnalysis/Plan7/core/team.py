import numpy as np
import collections

class Agent:
    def __init__(self, name, skill=0.5, learning_rate=0.01):
        self.name = name
        self.skill = skill
        self.learning_rate = learning_rate
        self.experience_count = 0

    def learn(self):
        self.experience_count += 1
        # 学習曲線: 経験を積むほどスキル向上 (Plan 1)
        self.skill = min(1.0, self.skill + self.learning_rate * (1 - self.skill))

class Team:
    def __init__(self, name, wip_limit=3, skill=1.0, num_agents=3, department: str | None = None, utilization_cap: float | None = None):
        self.name = name
        self.wip_limit = wip_limit
        self.skill = skill # チーム全体のベーススキル (Plan 5)

        # Ver2: 部署/稼働率上限（簡易）
        self.department = department
        # utilization_cap: 0..1 の比率。None は無制限扱い。
        self.utilization_cap = utilization_cap
        
        # 個人エージェントのリスト (Plan 1)
        self.agents = [Agent(f"{name}_Agent_{i}", skill=np.random.uniform(0.3, 0.7)) for i in range(num_agents)]
        
        self.inbox = collections.deque()
        self.wip = []
        self.completed_tasks = []
        
        # 組織間のエッジ情報 (delay, loss_prob, distortion_prob)
        self.outbound_edges = {}

    def add_edge(self, target_team_name, delay=1, loss_prob=0.1, distortion_prob=0.1):
        self.outbound_edges[target_team_name] = {
            'delay': delay,
            'loss_prob': loss_prob,
            'distortion_prob': distortion_prob
        }

    def receive_message(self, message):
        self.inbox.append(message)

    def send_message(self, message, target_team, current_time):
        edge = self.outbound_edges.get(target_team.name, {'delay': 0, 'loss_prob': 0, 'distortion_prob': 0})
        
        # チームの平均スキルにより劣化を抑制 (Plan 1 + Plan 5)
        avg_agent_skill = np.mean([a.skill for a in self.agents])
        # チーム全体のスキルとエージェントスキルの両方を考慮
        combined_skill = (self.skill + avg_agent_skill) / 2.0
        
        effective_loss_prob = edge['loss_prob'] * (1 - combined_skill * 0.5)
        effective_dist_prob = edge['distortion_prob'] * (1 - combined_skill * 0.5)

        # 遅延の設定
        arrival_time = current_time + edge['delay']
        
        # 欠落の判定
        if np.random.random() < effective_loss_prob:
            message['lost'] = True
            if 'metadata' in message:
                message['metadata'] = {} 
        
        # 誤解の判定
        if np.random.random() < effective_dist_prob:
            message['distorted'] = True
            if 'params' in message:
                message['params'] = {k: v * np.random.uniform(0.8, 1.2) for k, v in message['params'].items()}

        message['arrival_time'] = arrival_time
        target_team.receive_message(message)

    def handle_turnover(self, prob=0.01):
        """
        担当の入れ替わり（知識の流出） (Plan 1)
        """
        for i, agent in enumerate(self.agents):
            if np.random.random() < prob:
                # 新しい未熟なエージェントと交代
                self.agents[i] = Agent(f"{self.name}_NewAgent_{np.random.randint(1000)}", skill=0.3)

    def __repr__(self):
        dept = f", dept={self.department}" if getattr(self, 'department', None) else ""
        return f"Team({self.name}{dept}, skill={self.skill:.2f}, WIP={len(self.wip)}/{self.wip_limit})"
