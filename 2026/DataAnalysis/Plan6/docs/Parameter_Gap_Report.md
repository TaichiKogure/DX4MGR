# Plan 6 パラメータ詳細仕様書

本ドキュメントでは、Plan 6 の `Simulation.setup_with_params` に渡されるパラメータ辞書の詳細なデータ構造と、それぞれの値がシミュレーションロジックに与える影響を解説します。

---

## 1. データ構造の概要

パラメータは単一の辞書形式（`dict`）で管理されます。GUI (`gui_sim.py`) では、ユーザー入力をこの構造に変換してエンジンに渡します。

```python
params = {
    "scenario_id": "Plan6_Scenario_01",
    "steps": 500,
    "seed": 42,
    "strategy": "strategic",
    "use_digital_twin": True,
    
    # リソース設定
    "res_n_servers": 5,
    "proto_n_servers": 3,
    "mass_n_servers": 2,
    "wip_limit_res": 2,
    "wip_limit_proto": 1,
    
    # 技術閾値・動態
    "dr1_uncert_limit": 0.4,
    "dr2_matur_limit": 0.5,
    "dr3_matur_limit": 0.8,
    "decay": 0.7,
    "rework_load_factor": 0.5,
    "max_rework_cycles": 5,
    "tacitness": 0.5,
    
    # 通信・プロセス
    "loss_res_proto": 0.1,
    "dist_proto_ana": 0.05,
    "delay_ana_res": 2,
    "dr1_period": 30,
    "cost_per_review": 100,
    
    # 内部生成・拡張用
    "tech_items": [...],
    "teams": [...],
    "past_logs_file": "configs/past_logs.csv"
}
```

---

## 2. パラメータ詳細定義

### 2.1 シミュレーション制御
| キー | 型 | デフォルト | 内容 |
| :--- | :--- | :--- | :--- |
| `scenario_id` | `str` | - | シナリオの識別子。レポート名に使用。 |
| `steps` | `int` | 500 | シミュレーション実行期間（日）。 |
| `seed` | `int` | 42 | 乱数シード。再現性の確保に利用。 |
| `strategy` | `str` | "strategic" | エージェントのタスク選択戦略 ("strategic" or "random")。 |
| `use_digital_twin` | `bool` | True | 過去実績ログに基づいた技術成長モデルの有効化。 |

### 2.2 リソース・WIP管理
| キー | 型 | 内容 | ロジックへの影響 |
| :--- | :--- | :--- | :--- |
| `res_n_servers` | `int` | Research工程のエンジニア数。 | 同時実行可能な実験（Job）の数を決定。 |
| `proto_n_servers`| `int` | Prototype工程のエンジニア数。 | 試作フェーズのスループットを決定。 |
| `wip_limit_res` | `int` | Researchの仕掛品上限。 | これを超えると新規Jobが待ち行列（Queue）に入ります。 |

### 2.3 技術動態 (Plan 5x ロジック)
| キー | 型 | 内容 | ロジックへの影響 |
| :--- | :--- | :--- | :--- |
| `decay` | `float` | 知識減衰率。 | DR落ち等で時間が経過した際、Maturityが減少する速度。 |
| `tacitness` | `float` | 技術の暗黙知度。 | チーム間通信時の情報劣化（Loss/Distortion）を増幅。 |
| `rework_load_factor`| `float`| リワーク時の負荷軽減率。| 2回目以降の実験がどれだけ効率化されるか。 |

### 2.4 デザインレビュー (DR)
| キー | 型 | 内容 | ロジックへの影響 |
| :--- | :--- | :--- | :--- |
| `dr1_period` | `int` | DR1の開催周期（日）。 | 定期的なゲート審査のタイミング。 |
| `dr1_uncert_limit`| `float`| DR1通過に必要な不確実性。| これを下回らないとREWORK（やり直し）になります。 |
| `cost_per_review` | `float`| 1回のレビューにかかるコスト。| 開催されるたびに Total Gain から差し引かれます。 |

---

## 3. 動的構成機能 (Dynamic Config)

`core/simulation.py` は、上記の `params` に基づいて、内部的に `WorkGate` や `DRGate` を動的に生成します。

- **WorkGate**: `res_n_servers` や `wip_limit_res` を元に、リソース制約を持つ作業ノードが生成されます。
- **DRGate**: `dr1_period` や `dr1_uncert_limit` を元に、判定ロジックと周期的なイベントを持つ審査ノードが生成されます。

この動的構成により、将来的に JSON ファイル等から複雑な多段プロセスを柔軟に定義することが可能となっています。
