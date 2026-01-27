# DX4MGR Ver13 初心者ガイド

## 0. まずこれだけ
- Ver13は「固定DR日時 + 逆算工数割当 + 潜伏リスク(爆発)」を追加したモデルです。
- `scenarios.csv` を編集して `run.py` を実行すると、結果が `output/` に出ます。
- 1行=1シナリオ。1行目がBaseline扱いで比較の基準になります。

## 1. モデルの全体像 (ざっくり)
イメージは「製品企画の進め方」をシミュレーションする感じです。

- Small実験: アイデアの試作・小さな検証
- DR1: 小さな検証の結果を見て進めるか決める会議
- Mid実験: 中規模の検証 (プロトタイプ)
- DR2: 試作ラインでの判断 (潜伏リスクが顕在化)
- Fin実験: 最終検証
- DR3: 商品化判断

Ver13の追加ポイント:
- **DRCalendar**: DRの日時を固定カレンダーで管理 (dr*_periodから生成)
- **Scheduler**: 日次で工数を配分し **WorkPackage(計画)** を生成
- **LatentRisk**: 不確かさ/潜伏リスクがDR判定に補正として効く
- **Task** は実績、**WorkPackage** は計画 (AnyLogicっぽい分離)

## 2. フロー (固定構成)
```
SMALL_EXP -> BUNDLE_SMALL -> DR1 -> MID_EXP -> BUNDLE_MID -> DR2 -> FIN_EXP -> BUNDLE_FIN -> DR3 -> 完了
```

- **BUNDLE**: いくつかの案件をまとめてDRに送る仕組み。
  例: 6件たまったら1つに束ねてレビューに出す。
- **DR**: 固定日時で開かれる会議。容量(処理件数)を超えると次回に持ち越し。

## 3. 重要な指標の意味 (初心者向け)
- **Throughput (スループット)**: 1日あたり何件が完了したか。
- **Lead time (リードタイム)**: 案件が入ってから完了までの時間。
- **WIP**: 仕掛中の件数。多いほど渋滞が起きている。
- **Rework (差し戻し回数)**: DRで戻された回数。多いと手戻りが多い。
- **DR cost**: DRレビューのコスト合計 (感度比較用)。

例え: コンビニのレジ
- Throughputは「1日にさばいた客数」
- Lead timeは「並んでから会計が終わるまでの時間」
- WIPは「店内に滞留している客数」

## 4. シナリオを作る
`scenarios.csv` を編集します。

- 例: `arrival_rate=0.5` は「2日に1件の新規案件が来る」イメージ
- 例: `dr1_period=90` は「90日ごとにDR1会議がある」
- 例: `engineer_pool_size=10` は「計画上の稼働人数」

詳しい意味は `PARAMETER_GUIDE_Ver13.md` を見てください。

## 5. 実行
```bash
python3 2026/RDIssues/Ver13/run.py
```

出力先を変えたい場合は:
```bash
python3 2026/RDIssues/Ver13/run.py --out output_custom
```

シナリオCSVを指定したい場合は:
```bash
python3 2026/RDIssues/Ver13/run.py --scenarios 2026/RDIssues/Ver13/scenarios.csv
```

## 6. 出力の読み方 (主なファイル)
- `quality_gate_summary.csv`
  - 統計的に信用できるか、待ち時間が許容内かを判定した一覧
- `comparison_throughput.png`
  - スループットの比較 (95%信頼区間付き)
- `compare_violin.png`
  - 待ち時間分布の比較 (P90/P95の線あり)
- `compare_reworks.png`
  - 差し戻し回数の分布
- `ccdf_analysis.png`
  - 「待ち時間がx日を超える確率」(尾の重さを見る)
- `wip_time_series.png`
  - WIPの時間推移 (渋滞の増減)
- `gate_wait_heatmap.png`
  - ゲート別の平均待ち時間 (ボトルネック探し)
- `gate_wip_heatmap.png`
  - ゲート別の平均WIP
- `flow_time_breakdown.csv` / `flow_time_breakdown.png`
  - 工程ごとの平均時間の積み上げ
- `dr_cost_summary.csv`
  - DRレビューのコスト集計
- `job_details_*.csv`
  - 1件ごとの詳細ログ (待ち時間や摩擦の確認)
- `job_gantt_*.png`, `job_wait_heatmap_*.png`, `job_wait_dist_*.png`
  - サンプル案件のガント、時間帯ヒートマップ、待ち時間分布

## 7. まず試す調整 (例)
- 渋滞がひどい -> `dr*_period` を短く、`dr*_capacity` を増やす
- Mid/Finが詰まる -> `n_servers_mid` / `n_servers_fin` を増やす
- 差し戻しが多い -> `n_senior` を増やす / `conditional_prob_ratio` を下げる
- DR2で爆発が多い -> `engineer_pool_size` / `hours_per_day_per_engineer` を増やす
- まとめ待ちが長い -> `bundle_size_*` を小さくする

## 8. DOE (任意)
探索的に良いパラメータを探したい時は:
```bash
python3 2026/RDIssues/Ver13/run_doe.py
```

`output_doe/` に感度分析結果と、上位シナリオ候補が出力されます。

## 9. よくある困りごと
- **完了ジョブが0件**:
  - `days` を増やす、`arrival_rate` を上げる、DR周期を短くする
- **結果が不安定**:
  - `n_trials` を増やして分布を安定させる
- **DR2のNOGOが急増**:
  - `engineer_pool_size` を増やす / `dr2_period` を長くして準備時間を確保
