# DX4MGR Ver13 パラメーターガイド

## 1. 使い方の前提
- `scenarios.csv` の1行が1シナリオです。
- 1行目がBaselineとして比較の基準になります。
- 単位は「日」が基本です。

例:
- `arrival_rate=0.5` -> 平均2日に1件
- `dr1_period=90` -> 90日ごとにDR1会議

## 2. パラメータ一覧 (カテゴリ別)

### 2.1 シナリオ制御
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| scenario_name | シナリオ名 | 例: `01_Baseline` |
| n_trials | モンテカルロ試行回数 | 大きいほど安定するが遅い |
| days | シミュレーション期間 | 長いほど完了が増えやすい |

### 2.2 到着と作業時間
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| arrival_rate | 到着率(件/日) | 0.5なら平均2日に1件 |
| small_exp_duration | Small実験の平均日数 (指数分布) | 大きいほどSmallが長くなる |
| mid_exp_duration | Mid実験の平均日数 (指数分布) | Mid工程の遅さに直結 |
| fin_exp_duration | Fin実験の平均日数 (指数分布) | Fin工程の遅さに直結 |

### 2.3 バンドル設定
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| bundle_size_small | SmallからDR1に出す束のサイズ | 小さいと待ちは減るがDR負荷増 |
| bundle_size_mid | MidからDR2に出す束のサイズ | 同上 |
| bundle_size_fin | FinからDR3に出す束のサイズ | 同上 |

### 2.4 DRカレンダー・容量・コスト
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| dr1_period / dr2_period / dr3_period | 会議の開催間隔(日) | Ver13ではDRCalendar生成に使用 |
| dr_capacity | DR共通の容量 (未指定時の既定) | dr1/2/3が空なら使う |
| dr1_capacity / dr2_capacity / dr3_capacity | DRごとの処理上限 | 大きいほど詰まりにくい |
| dr_quality | DR品質の上書き(0..1) | 指定時は承認者構成より優先 |
| dr1_cost_per_review / dr2_cost_per_review / dr3_cost_per_review | DR1件あたりのコスト | `dr_cost_summary.csv` に反映 |
| decision_latency_days | 会議後の意思決定遅延 | 大きいほどリードタイム増 |

補足:
- DRの日時は `dr*_period` から固定スケジュールを生成しています。
- 不規則なDR日時を使いたい場合は `runner/adapters.py` の `DRCalendar` を編集してください。
- `dr_quality` が空/NaN の場合は承認者構成から算出した品質を使用します。

### 2.5 差し戻し (Rework)
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| rework_load_factor | 差し戻し重み->増殖の倍率 | 大きいほど負荷増 |
| dr2_rework_multiplier | DR2だけ倍率を掛ける | DR2だけ厳しくする時に有効 |
| max_rework_cycles | 追加差し戻しの上限 | 例: 5なら6回目以降は増殖なし |
| decay | 差し戻し回数に伴う減衰率 | 小さいほど繰り返しは弱まる |
| rework_beta_a / rework_beta_b | 重み分布(Beta分布)の形 | a,bで偏りが変わる |
| rework_task_type_mix | 追加タスクの再投入比率 | SMALL_EXPへ戻す割合(0..1) |
| conditional_prob_ratio | 条件付き判定の割合 | 1.0でNOGOがほぼ出ない |

補足:
- `rework_task_type_mix` の比率分だけ、増殖タスクを新規ジョブとしてSMALL_EXPへ再投入します。

### 2.6 承認者構成 (DR品質と容量)
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| n_senior | Senior人数 | 品質と容量が高い |
| n_coordinator | Coordinator人数 | 中程度 |
| n_new | New人数 | 品質と容量が低い |

品質と容量は人数から自動計算されます。
例: Senior 2名 + Coordinator 1名 -> 容量 = 2*7 + 1*3 = 17
`dr_quality` が指定されている場合は、品質の自動計算を上書きします。

### 2.7 Mid/Finの並列サーバ数
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| n_servers_mid | Mid実験の並列数 | 大きいほど処理は速いが摩擦増 |
| n_servers_fin | Fin実験の並列数 | 同上 |

Small実験の並列数は固定(999)で、パラメータはありません。

### 2.8 摩擦 (コミュニケーションロス)
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| friction_model | 摩擦モデル | `linear` または `pairs` |
| friction_alpha | 摩擦の強さ | 大きいほど遅くなる |

### 2.9 Scheduler / LatentRisk
| パラメータ | 意味 | 例・影響 |
| --- | --- | --- |
| engineer_pool_size | 計画上のエンジニア人数 | 大きいほどWorkPackageが増える |
| hours_per_day_per_engineer | 1人あたりの計画工数/日 | 大きいほどLatentRiskが下がりやすい |

補足:
- Schedulerは日次(SCHED_TICK)で動き、WorkPackageを生成してLatentRiskを更新します。
- 計画対象はキュー内だけでなく稼働中ジョブも含みます。
- `tick_days` や `LatentRisk.scale_factor` は現状固定値で、変更する場合はコード側で調整します。

### 2.10 予約・現状未使用の項目
| パラメータ | 状態 | 補足 |
| --- | --- | --- |
| sampling_interval | 任意 | `simulate_standard_flow` に渡せばWIPの記録間隔を変更可能 |

## 3. 具体例 (Baselineの読み方)
`arrival_rate=0.2` の場合、平均5日に1件の到着です。
`bundle_size_small=6` なら、6件集まるまでDR1には送られません。
`dr1_period=90` なら、90日ごとにDR1会議が開かれます。

## 4. よく使う調整パターン
- **会議の待ちを減らす**: `dr*_period` を短く、`dr*_capacity` を増やす
- **バンドル待ちを減らす**: `bundle_size_*` を小さく
- **差し戻しを減らす**: `n_senior` を増やす / `conditional_prob_ratio` を下げる
- **処理能力を上げる**: `n_servers_mid` / `n_servers_fin` を増やす
- **DR2爆発を抑える**: `engineer_pool_size` / `hours_per_day_per_engineer` を増やす
