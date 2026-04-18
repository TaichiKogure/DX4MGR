Plan7「Completed Jobs」KPIが変化しない件の調査レポート（原因と感度）

作成日: 2026-03-28

## 要約（結論）
- 現行Plan7では、`Completed Jobs` は「シミュレーション期間内に終端へ到達したジョブ数（= 完了ジョブ配列の長さ）」として計上されています。
- 既定シナリオは「初期投入ジョブ数（デフォルト3件）」のみを処理対象としており、`steps`（時間地平）が十分に長い場合、最終的に全件が完了に到達する設計です。そのため、サーバ数・DR会議周期・閾値など多くのパラメータを変えても、完了件数は一定（= 初期投入数）になりやすい、というのが不変の主原因です。
- `Completed Jobs` を変動させる主な要因は、(1) 初期投入ジョブ数、(2) シミュレーション地平（`steps`）が不十分で完了に至らなかった件数、に限られます。その他のパラメータは主として「所要時間（スループット/リードタイム）」へ影響し、完了件数そのものには影響しにくい構造です。

## 参照コード（根拠）
- KPI定義と集計:
  - `2026/DataAnalysis/Plan7/core/simulation.py`
    - KPI初期化: 行29-43
    - 期間終了時の集計: 行494-505（`self.kpis['completed_jobs'] = len(self.engine.results["completed_jobs"])`）
- 完了の記録トリガ:
  - `Plan7MeetingGate.process`（DRゲート）: 同ファイル 行318-382
    - `target` が無い（フロー終端。既定では DR3 の GO 判定）→ `self.engine.results["completed_jobs"].append(job)`
  - `SimulationEngine.handle_event`（終端への ARRIVAL も完了扱い）:
    - `2026/DataAnalysis/Plan7/core/engine.py` 行72-82
- 初期投入ジョブ数と投入方法:
  - `simulation.py` 行472-480
    - `num_initial_jobs`（デフォルト3件）を `start_node` へ ARRIVAL で投入

## 現象の整理
- DOEやGUI比較で、`res_n_servers` や `dr*_period`、会議品質、SLA/ハンドオフ等を変えても `Completed Jobs` が変わらない事象を確認。
- 上記の通り、到着ジョブ数が固定（=3件）で、十分な時間（`steps` が大きい）を与えると、再作業（rework）を繰り返しても最終的に GO に到達し「全件完了」となるため、完了件数は不変となります。

## パラメータ感度（何が効き、何が効かないか）
- 完了件数に直接効く（一次効果が強い）
  - `num_initial_jobs`（初期投入ジョブ数）: 完了上限を直接規定。線形に効きます。
  - `steps`（シミュレーション地平・最大日数）: 不足すると「未完了」が残り、`Completed Jobs` が低下。閾値的に効きます。
- 条件付きで効く（地平が限られる場合のみ間接的に効く）
  - DR関連（`dr*_period`, `approvers.capacity`, `approvers.quality`, `thresholds`）: 会議頻度/キャパ/合否確率が低いと完了までの時間が延び、短い `steps` では未完了が増える→間接的に完了件数を押し下げ。
  - 工程並列度（`*_n_servers`）: 並列度が低いと待ちが伸び、短い `steps` 下での完了件数が減る可能性。
  - ハンドオフ/部門考慮（`consider_handoffs`, `consider_departments`, handoff `transfer_time_dist`, `q_if`, `info_loss_lambda`）: 遅延・品質劣化によりリワークや所要時間が増え、短い `steps` では未完了が増える可能性。
- ほとんど効かない/効きにくい（時間無限大に近い条件では影響希薄）
  - DR/工程パラメータ全般: 充分に長い地平では、再作業を経て最終的に GO に到達しうるため、完了件数は初期投入数に収束しやすい。
  - `rework_policy.max_rework_cycles` 等: 現行の `Plan7MeetingGate.process` は簡易な再投入（直接 ARRIVAL）で、`rework_policy` の制約が実ロジックに強く反映されていません（将来拡張の余地）。
  - SLA/コスト係数: KPIの `sla_violations`, `dept_cost_time` には効くが、完了有無には原則間接のみ。

## 模範となる比較シナリオ（A/B/C）
- シナリオA: ベースライン（完了件数が不変になる代表例）
  - 入力例: `num_initial_jobs=3`, `steps=365`, `res_n_servers=5`, `proto_n_servers=3`, `dr1_period=14`（他は既定）
  - 期待: `Completed Jobs ≈ 3`（全件完了）。`throughput`/`lead_time` はパラメータで変化するが、完了件数は不変。
- シナリオB: 時間制約で感度を出す
  - 入力例: シナリオAから `steps=60` に短縮。`dr1_period` を 7/14 に切替、`res_n_servers` を 3/7 に変更して比較。
  - 期待: 地平が短いため、DRの頻度や並列度の差が「期間内完了件数」に顕在化。`Completed Jobs` が条件に応じて 0〜3 の範囲で変動。
- シナリオC: 入力ボリュームで感度を出す
  - 入力例: `num_initial_jobs=30`, `steps=120`。`proto_n_servers` を 2/6、`dr2_period` を 21/42 で比較。
  - 期待: ボトルネックの差が「期間内に捌けた件数」に反映され、`Completed Jobs` が大きく変動。並列度や会議頻度の効果が明瞭。

## 再現手順（最小セット）
1. ベースライン（A）
   - コマンド例（既存DOEランナー）:
     - `python 2026/DataAnalysis/Plan7/experiments/plan7_doe.py --steps 365 --seeds 3 --outdir 2026/DataAnalysis/Plan7/reports/doe`
   - 期待観察: `runs.csv` 中の `completed_count`（計測メトリクス）や KPI `completed_jobs` がグリッド間でほぼ一定。
2. 時間制約（B）
   - `--steps 60` に短縮して同様に実行。`res_n_servers`, `dr1_period` を含むデフォルトグリッドで比較。
   - 期待観察: `completed_count`/KPI `completed_jobs` に差が出る。
3. 入力ボリューム（C）
   - `base_params` に `num_initial_jobs: 30` を追加した JSON を用意し、`--config` で指定して実行。
   - 期待観察: 並列度・会議周期の差が `completed_count`/`completed_jobs` に顕在化。

## 改善提案（必要に応じて）
- 分析/KPI設計の観点
  - `Completed Jobs` を主目的変数にする場合、「時間地平（期間内完了数）」とセットで設計する（= スループット指標との組）。
  - 追加KPI案: `completion_rate = completed / arrivals`、`on_time_completion`（SLA内での完了件数）、`no_go_termination`（NO GOで打切り件数）など。
- モデル/実装の観点
  - `rework_policy`（再作業回数の上限・減衰）の実ロジック反映を強化し、無限再挑戦で最終的に必ず GO へ到達する状態を避ける。
  - 到着プロセスを拡張（`arrival_rate`/バッチ到着など）し、入力ボリューム変化で `Completed Jobs` が自然に動くようにする。
  - DRの `nogo_node_id` を終端（打切り）に向ける選択肢を追加し、「NO GO = 完了しない」が件数に反映される選択肢を用意。

## 備考
- 既存DOEレポートの設計では、可視化・主効果は主に `throughput` や `lead_time` に焦点があり、`Completed Jobs` は（地平が十分な限り）安定的です。期間内完了を評価軸とする場合は、上記シナリオB/Cのように「短い地平」または「投入ジョブ増」を組み合わせると感度が出ます。