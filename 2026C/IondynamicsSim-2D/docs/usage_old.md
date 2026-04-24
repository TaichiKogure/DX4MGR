# IondynamicsSim 使用方法

IondynamicsSimは、リチウムイオン電池の正極内におけるイオン輸送と濃度分布をシミュレーション・分析するためのツールです。特に「厚膜電極」における課題の可視化とKPI算出に特化しています。

## 1. 実行方法

### GUIによる実行
1. `run_gui.bat` をダブルクリックしてGUIを起動します。
2. 左側のパネルでパラメータ（厚み、空隙率、Cレートなど）を調整します。
3. `Run Simulation` ボタンを押すとシミュレーションが実行されます。
4. 実行完了後、各種プロットボタンが有効になります。
5. `Case Comparison` セクションで、特定のパラメータ（厚みなど）を振った比較計算が可能です。

### CLIによる実行
コマンドプロンプトやPowerShellから以下のコマンドを実行できます。

**単体実行:**
```bash
python -m iondynamics.cli run --config configs/default.yaml
```

**比較レポート作成:**
```bash
# 厚みを 60, 80, 100 um で比較
python -m iondynamics.cli compare --config configs/default.yaml --axis thickness --values 60 80 100
```

**アニメーション作成:**
```bash
python -m iondynamics.cli animate --config configs/default.yaml
```

## 2. 主要KPI（重要指標）の解説

Phase 1で追加された以下の指標により、厚膜化による悪化を定量化します。

*   **Δc_e (Electrolyte Concentration Difference):** セパレータ側と集電体側の電解液濃度差。値が大きいほど、液内の輸送が追いついていないことを示します。
*   **Max Concentration Gradient:** 厚み方向の最大濃度勾配。局所的な負荷の集中度合いを判定します。
*   **Effective Ionic Resistance Index:** 濃度勾配に基づいた見かけの輸送抵抗指標。厚膜や低空隙率で値が増大します。
*   **Depletion Onset Time:** 電解液が枯渇（濃度が閾値以下に到達）するまでの時間。枯渇しない場合は -1 と表示されます。
*   **Ce Ratio:** セパレータ側濃度 / 集電体側濃度。

## 3. 出力結果の確認

実行結果は `outputs/runs/YYYYMMDD_HHMMSS_[CaseName]` フォルダに保存されます。

*   `results.csv`: 時系列データ（電圧、電流、各KPI）
*   `summary.csv`: 最終値や平均値をまとめたサマリ
*   `thickness_ce_dist.csv`: 厚み方向の濃度分布データ
*   `voltage_time.png`: 電圧推移グラフ
*   `thickness_ce_profiles.png`: 厚み方向の濃度分布プロファイル
*   `kpi_time_series.png`: 主要KPIの時系列推移
*   `report.md`: (比較実行時) 複数ケースを比較したMarkdownレポート

## 4. 設定ファイルのカスタマイズ

`configs/default.yaml` をコピーして編集することで、詳細なシミュレーション条件を設定できます。
特に `pybamm` セクションでは、PyBaMMの標準パラメータセットの選択や、個別の物性値の上書きが可能です。
