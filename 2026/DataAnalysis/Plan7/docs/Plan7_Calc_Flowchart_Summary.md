# Plan7 計算フローチャート 要約

本ページは、Plan7 の計算フローを短時間で把握できるようにまとめた要約版です。詳細な図解・説明は「計算フローチャート（詳細）」を参照してください。

---

## 目的と前提
- 対象: 2026/DataAnalysis/Plan7（現行Plan7）
- 対応モジュール: `gui_sim.py`, `core/simulation.py`, `core/engine.py`, `analysis/metrics.py`, `core/scenario.py`

## 全体像（要点）
- GUIの実行種別に応じてスレッドを起動し、入力パラメータから `Simulation.setup_with_params` で技術・チーム・ゲートを構築。
- `SimulationEngine.run` が ARRIVAL/PROCESS_READY/WORK_COMPLETE/MEETING_START のイベントを時系列で処理。
- 完了後に `analysis/metrics.calculate_metrics` でKPIやDR統計、WIP指標を集計。
- `ScenarioManager.visualize_single` が各シナリオのレポート画像（summary/tech/wip）を生成。GUIの「詳細レポート出力」は直近比較の全シナリオ分を自動一括出力。

## 実行モード別の流れ
1) 単体実行（`run_simulation_thread`）
   - パラメータ整形 → `Simulation`生成 → `setup_with_params` → `engine.run(steps)`。
   - 実行中は一定間隔で進捗・部分結果（WIP/技術スナップショット）をGUIに通知。
   - 完了後にメトリクス算出→GUI表示→必要に応じて詳細レポート出力。

2) 3シナリオ比較（`run_comparison_thread`）
   - 3本のシナリオを並行実行、部分結果で比較ダッシュボードを段階更新。
   - 完了時に各シナリオのKPI/metricsを保持。「詳細レポート出力」で全シナリオPNGを一括生成。

3) DOE一括（`run_batch_thread`）
   - 基準パラメータから特定要素（例: ResEng人数）を振って複数回実行。
   - まとめてCSV（`reports/batch_results.csv`）に保存。最後の実行結果をGUIに反映。

## DESエンジンの動き（`SimulationEngine.run`）
- 初期イベント投入 → 最小時刻イベントから順に処理するイベント駆動モデル。
- 主なイベントと処理:
  - ARRIVAL: ノードへ投入、条件が整えば `PROCESS_READY` を予定。
  - PROCESS_READY: ノード `process(now)` を実行（作業/会議）。
  - WORK_COMPLETE: 作業完了。次工程へ `ARRIVAL` か完了集計へ。
  - MEETING_START: DR会議の判定処理。
- 一定間隔でWIPをサンプリング（`results.wip_history`）。

## DR会議（Plan7MeetingGate）の判定
- 会議で対象案件を取り出し、平均不確実性/成熟度を集計。
- 閾値達成度に応じて通過基準品質qを調整し、乱数で GO / COND / NO_GO を決定。
- 結果は通過率や待ち時間、意思決定遅延の統計に反映。

## WorkGate（Plan7WorkGate）の処理
- ジョブ取得 → 対象技術選択（phase依存） → 担当チーム決定。
- 伝達劣化（ロス/歪み、tacitness依存）を反映 → `Experiment.run()` → 学習（`agents.learn`）。
- KPI更新後、処理時間をサンプリングし `WORK_COMPLETE` をスケジュール。

## メトリクス（`analysis/metrics.py`）
- 入力: `completed_jobs`, `nodes_stats`, `engine.now`, `wip_history`。
- 出力例:
  - 概要: スループット、リードタイム（p50/p90/p95）、差し戻し回数、平均WIP。
  - DR統計: サイクル/待ち/意思決定遅延、GO/COND/NO_GO比率。
  - WIP: 総/ノード別平均、時系列ヒストリー（ヒートマップ用）。
  - CCDFやRaw値（詳細可視化の元データ）。

## 可視化・レポート出力
- `ScenarioManager.visualize_single(sim, metrics, scenario_id)` で `reports/viz/*` にPNG生成。
- 例: `Scenario_X_summary.png`（主要KPIまとめ）、`_tech.png`（技術推移）、`_wip.png`（WIPヒートマップ）。
- GUIの「詳細レポート出力」で、直近の比較結果やバッファから全シナリオを自動列挙し一括出力。

## 入出力と再現のポイント
- 入力例: `scenario_id`, `steps`, `strategy`, `seed`、リソース/DR設定、伝達ロス、Rework方針など。
- 出力: `reports/viz/*.png`, `reports/batch_results.csv` とGUIログ。
- 再現性: Digital Twin有効時は `configs/past_logs.csv`、乱数 `seed` で再現担保。

## 典型シーケンス（単体実行）
1) GUIでパラメータ入力→実行
2) `setup_with_params` で構成生成（技術/チーム/ゲート/Rework）
3) 初期イベント投入（ARRIVAL/会議予約）
4) `engine.run(steps)` でイベントを時系列処理
5) サンプリングごとにWIP記録とGUI更新
6) 完了後 `calculate_metrics()` → GUI表示
7) 必要に応じて詳細レポート出力（PNG一式）

## 参考リンク
- 計算フローチャート（詳細）: [Plan7_Calc_Flowchart.md](Plan7_Calc_Flowchart.md)

---
最終更新: 2026-03-25
