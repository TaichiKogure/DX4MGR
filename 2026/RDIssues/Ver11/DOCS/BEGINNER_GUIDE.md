# DX4MGR Ver11 初心者ガイド

DX4MGR（Digital Transformation for Manager） Ver11へようこそ。
このツールは、研究開発や設計業務のような「不確実性が高く、差し戻し（Rework）が発生しやすい業務」をシミュレートし、最適な改善策を統計的に見つけるためのツールです。
Ver11では、リソース（要員構成）の能力差や、差し戻しによる「タスクの増殖」など、より現実に近い挙動をモデル化しています。

## 解析の流れ

### 1. 問いを立てる
まずは何を改善したいか決めましょう。
- 「審査周期を短縮したら、スループットは向上するか？」
- 「シニア層を重点配置した場合、リードタイムはどう変化するか？」

### 2. シナリオを設定する (`scenarios.csv`)
- dr_periods: 例 `30|65` で多段DRを指定（空なら dr_period を使用）
`scenarios.csv` を編集して、比較したい条件を書き込みます。
各パラメータの意味は `PARAMETER_GUIDE_Ver10.md` を参照してください。

### 3. シミュレーションを実行する
ターミナルを開き、以下のコマンドを入力します。
```bash
python3 2026/RDIssues/Ver11/run.py
```

### 3.5 DOE探索（任意）
探索用は `run_doe.py` を使い、DOEの結果からシナリオ候補を作成します。
```bash
python3 2026/RDIssues/Ver11/run_doe.py
```
`output_doe/` に `doe_results_scored.csv` と `scenarios_from_doe.csv` が出力されます。

### 4. 結果を読み解く
`output/` フォルダに生成されるグラフを確認します。
- **Job進行（サンプルガント）**: `job_gantt_...png`
- **ゲート別待ち時間（時間帯ヒートマップ）**: `job_wait_heatmap_...png`
- **ゲート別待ち時間分布**: `job_wait_dist_...png`
- **Quality Gate判定一覧(CSV)**: `quality_gate_summary.csv`
  - PASS（緑）が出ていれば、そのシナリオは目標基準（スループットや待ち時間）をクリアしています。
- **ボトルネックはどこか？** (`gate_wait_heatmap.png`, `gate_wip_heatmap.png`)
  - どの工程で待ち時間が発生しているか一目でわかります。
- **本当に効果があるか？** (`comparison_throughput.png`)
  - 統計的な有意差（改善率）を確認できます。

## 困ったときは
- **「ModuleNotFound...」が出る**: 必要なライブラリ（pandas, numpy, matplotlib, scipy）がインストールされているか確認してください。
- **実行エラー**: スクリプトのフルパスを使って実行するか、Ver11 ディレクトリに移動してから実行してください。
