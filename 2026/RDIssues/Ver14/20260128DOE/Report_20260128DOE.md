### Ver14 差し戻し有効域探索レポート (20260128DOE)

#### 1. 実施概要
- 参照: 20260127DOE2 の Baseline_v2 設定を基準に、差し戻し関連パラメータを中心に探索。
- サンプル数: 30 / 試行回数: 2 (Monte Carlo)
- 評価基準: Throughput >= 0.000500 を満たし、rework_completed_countが一定以上の領域を『差し戻し有効域』と定義
- 追加条件: rework_reinject_mode は `ratio` に固定（rework_reinject_ratio を探索対象化）

#### 2. Baseline_v2 の参考値（平均）
- Throughput: 0.000137
- Completed(Primary): 0.50
- Rework Completed: 12.50
- Rework Throughput: 0.003425
- Rework LT P90: 48.77

#### 3. 差し戻し有効域（Q10–Q90の目安）
| パラメータ | 有効域(10–90%) |
| --- | --- |
| rework_load_factor | 0.863 – 2.828 |
| dr2_rework_multiplier | 1.213 – 3.232 |
| decay | 0.535 – 0.885 |
| rework_beta_a | 1.914 – 4.460 |
| rework_beta_b | 2.349 – 8.873 |
| rework_reinject_ratio | 0.526 – 0.882 |
| conditional_prob_ratio | 0.540 – 0.920 |
| dr_quality | 0.516 – 0.830 |
| decision_latency_days | 1.650 – 12.078 |
| adoption_rate | 0.466 – 0.834 |

#### 4. 上位サンプル（差し戻し完了数が多い順）
| rework_completed_count | throughput | rework_ratio | rework_load_factor | dr2_rework_multiplier | decay | rework_beta_a | rework_beta_b | rework_reinject_ratio | conditional_prob_ratio | dr_quality | decision_latency_days | adoption_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 26.50 | 0.001096 | 0.869 | 2.064 | 3.136 | 0.509 | 4.684 | 8.295 | 0.535 | 0.928 | 0.543 | 11.481 | 0.490 |
| 25.50 | 0.002740 | 0.718 | 3.026 | 2.134 | 0.719 | 4.344 | 2.501 | 0.949 | 0.634 | 0.833 | 7.520 | 0.415 |
| 19.50 | 0.000822 | 0.867 | 0.846 | 1.193 | 0.889 | 4.312 | 4.831 | 0.524 | 0.888 | 0.314 | 3.264 | 0.762 |
| 19.00 | 0.001370 | 0.792 | 1.594 | 1.982 | 0.567 | 1.832 | 4.597 | 0.768 | 0.812 | 0.571 | 6.149 | 0.632 |
| 17.00 | 0.002055 | 0.694 | 2.761 | 1.044 | 0.728 | 3.708 | 3.937 | 0.765 | 0.797 | 0.767 | 4.040 | 0.703 |

#### 5. 解釈メモ
- Ver14はDR固定日が1回のみのため、Throughputは小さくなりがち。差し戻しの有効性は「一次完了が極小でも rework 完了が発生するか」で評価。
- rework_load_factor / dr2_rework_multiplier を上げると rework 完了数が増えやすいが、条件付き率や品質のバランスが崩れると一次完了が消失しやすい。
- adoption_rate が低すぎると上流でRejectされ、差し戻し流量が細るため注意。

#### 6. 出力ファイル
- doe_rework_results.csv: DOE全サンプルの結果
- doe_rework_effective.csv: 有効域フィルタ後のサンプル
- effective_ranges.json: 有効域レンジ(Q10–Q90)
- run_summary.json: 実行設定とBaseline要約
