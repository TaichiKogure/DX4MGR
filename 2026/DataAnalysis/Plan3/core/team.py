import numpy as np
import collections

class Team:
    def __init__(self, name, wip_limit=3, skill=1.0):
        self.name = name
        self.wip_limit = wip_limit
        self.skill = skill
        
        self.inbox = collections.deque()
        self.wip = []
        self.completed_tasks = []
        
        # 組織間のエッジ情報 (delay, loss_prob, distortion_prob)
        # key: target_team_name, value: dict
        self.outbound_edges = {}

    def add_edge(self, target_team_name, delay=1, loss_prob=0.1, distortion_prob=0.1):
        self.outbound_edges[target_team_name] = {
            'delay': delay,
            'loss_prob': loss_prob,
            'distortion_prob': distortion_prob
        }

    def receive_message(self, message):
        self.inbox.append(message)

    def process_step(self, current_time):
        """
        1ステップ（1日等）の処理
        """
        # WIPが空いていればinboxから取り出す
        while len(self.wip) < self.wip_limit and self.inbox:
            task = self.inbox.popleft()
            task['start_time'] = current_time
            self.wip.append(task)
            
        # タスクの進行（ここでは単純化して一定期間で終わるか、リソース待ち等にする）
        # 実装の詳細はシミュレーター側で制御
        pass

    def send_message(self, message, target_team, current_time):
        edge = self.outbound_edges.get(target_team.name, {'delay': 0, 'loss_prob': 0, 'distortion_prob': 0})
        
        # チームのスキルにより劣化を抑制
        effective_loss_prob = edge['loss_prob'] * (1 - self.skill * 0.5)
        effective_dist_prob = edge['distortion_prob'] * (1 - self.skill * 0.5)

        # 遅延の設定
        arrival_time = current_time + edge['delay']
        
        # 欠落の判定
        if np.random.random() < effective_loss_prob:
            message['lost'] = True
            # メタデータの欠落などを表現
            if 'metadata' in message:
                message['metadata'] = {} 
        
        # 誤解の判定
        if np.random.random() < effective_dist_prob:
            message['distorted'] = True
            # 内容の変質を表現
            if 'params' in message:
                message['params'] = {k: v * np.random.uniform(0.8, 1.2) for k, v in message['params'].items()}

        message['arrival_time'] = arrival_time
        target_team.receive_message(message)

    def __repr__(self):
        return f"Team({self.name}, WIP={len(self.wip)}/{self.wip_limit}, Inbox={len(self.inbox)})"
