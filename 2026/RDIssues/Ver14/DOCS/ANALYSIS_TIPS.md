# DX4MGR Ver13 分析のコツ

## 1. まず見る順番 (迷ったらこれ)
1) `quality_gate_summary.csv`
   - PASS/FAILで「統計的に信用できるか」を確認。
2) `comparison_throughput.png` と `compare_violin.png`
   - スループットと待ち時間(P90)のバランスを確認。
3) `gate_wait_heatmap.png` と `flow_time_breakdown.png`
   - どの工程がボトルネックかを特定。

## 2. 症状別の原因と手当て

### スループットが低い
- 原因候補: DRの会議頻度が低い / 容量が足りない / バンドル待ちが長い
- 手当て例:
  - `dr1_period`〜`dr3_period` を短くする
  - `dr1_capacity` 〜 `dr3_capacity` を増やす
  - `bundle_size_*` を小さくする

### P90待ち時間が長い (尻尾が重い)
- 原因候補: バンドル待ちが長い / DRで滞留 / ある一部の工程が極端に遅い
- 手当て例:
  - `bundle_size_*` を小さくする
  - `decision_latency_days` を減らす
  - `n_servers_mid` / `n_servers_fin` を増やす

### WIPが高い (渋滞)
- 原因候補: 入ってくる量に対して処理能力が不足
- 手当て例:
  - `arrival_rate` を下げる
  - DR容量やMid/Finのサーバ数を増やす

### 差し戻し回数が多い
- 原因候補: DRの品質が低い / 条件付き判定が多い
- 手当て例:
  - `n_senior` を増やし、`n_new` を減らす
  - `conditional_prob_ratio` を下げる
  - `rework_load_factor` を下げる

### DR2でNOGOが増える (潜伏リスク爆発)
- 原因候補: LatentRiskの evidence が不足
- 手当て例:
  - `engineer_pool_size` / `hours_per_day_per_engineer` を増やす
  - `dr2_period` を長くして準備期間を確保
  - `arrival_rate` を下げて計画余力を作る

### DRコストが高い
- 原因候補: DRが多すぎる / バンドルが小さすぎる
- 手当て例:
  - `bundle_size_*` を上げる
  - `dr1_period` 〜 `dr3_period` を少し長くする
  - `dr*_cost_per_review` を見直す (比較用の感度調整)

## 3. ボトルネック特定のコツ
- `flow_time_breakdown.png` で棒が長い工程が「時間を食っている」箇所。
- `gate_wait_heatmap.png` で赤い箇所は「待ちが長い」箇所。
- 2つが同じ工程を指していれば、そこが真のボトルネック。

例:
- BUNDLE_SMALLが長い -> `bundle_size_small` を下げる
- DR2待ちが長い -> `dr2_period` を短く、`dr2_capacity` を増やす

## 4. バンドル調整の考え方
- **小さくする**: 待ち時間は減るが、DRの処理回数が増える
- **大きくする**: DR負荷は減るが、まとめ待ちが増える

例え: レジのまとめ会計
- 1人ずつ会計 -> 早いが件数が増える
- 10人まとめて会計 -> レジ回数は減るが待ち行列が伸びる

## 5. サーバ数と摩擦の罠
`n_servers_mid` / `n_servers_fin` を増やすと処理は速くなる一方、
摩擦係数 `friction_alpha` が大きいと逆に遅くなることがあります。

- `friction_model=linear` の場合:
  - 実効時間 = 基本時間 * (1 + friction_alpha * (n_servers - 1))

「人を増やしたのに遅くなる」場合は `friction_alpha` を下げるのが有効です。

## 6. ベースライン比較のコツ
- `scenarios.csv` の1行目がBaseline扱いです。
- 改善率は「信頼区間が重ならないか」を簡易判定しています。
- 少ない `n_trials` だとブレが大きいので、結論を急がないのがコツ。

## 7. 追加で深掘りしたいとき
- `job_details_*.csv` で個別案件の待ち時間や摩擦を確認
- `job_gantt_*.png` で「どこで止まっているか」を目視
- `ccdf_analysis.png` で長期滞留がどれくらいあるか確認
