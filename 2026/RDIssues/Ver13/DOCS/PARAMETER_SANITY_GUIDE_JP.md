# Ver13 パラメータ勘所表（破綻しないスケール目安）

## 前提
- 単位は「日」。
- 破綻 = ほぼ未完了・WIP爆増・待ち時間が支配的・DR待ちが無限に伸びる状態。
- 目安レンジは **Baseline (scenarios.csv)** を中心にした実務的な安全域。
- 連動関係（目安式）は下段の「簡易キャパ判定」を参照。

---

## 1) 目安レンジ（感度確認向け）

### シナリオ制御
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| n_trials | 20〜200 | 少なすぎるとばらつき大、300超は重い |
| days | 365〜1500 | 200未満は完了少、2000超は重い |

### 到着と作業時間
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| arrival_rate | 0.05〜0.5 | 容量に対して高すぎるとWIP増・LT発散 |
| small_exp_duration | 3〜14 | 20超でSmall滞留が強くなる |
| mid_exp_duration | 10〜40 | 50超でMidがボトルネック化 |
| fin_exp_duration | 10〜60 | 80超でFin滞留が顕著 |

### バンドル
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| bundle_size_small | 2〜10 | 大きすぎるとDR1が開いても束不足 |
| bundle_size_mid | 2〜6 | 大きすぎるとDR2待ちが長期化 |
| bundle_size_fin | 2〜6 | 大きすぎるとDR3が動かない |

### DRカレンダー・容量・品質
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| dr1/2/3_period | 30〜120 | 180超はDR待ちが支配的になる |
| dr1/2/3_capacity | 5〜30 | arrivalに比べて低いとDR詰まり |
| dr_quality | 0.4〜0.9 | 1.0近いと速度低下で詰まりやすい |
| dr_quality_speed_alpha | 0.5〜2.0 | 3超で容量補正が極端になりやすい |
| decision_latency_days | 0〜14 | 30超はリードタイム支配要因になる |

### 差し戻し (Rework)
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| rework_load_factor | 0.3〜2.0 | 3超で再作業が爆発しやすい |
| dr2_rework_multiplier | 0.5〜3.0 | 4超でDR2起因の増殖が激化 |
| max_rework_cycles | 2〜6 | 8超は計算重い・過剰ループ | 
| decay | 0.5〜0.9 | 0.95超で収束しづらい |
| rework_beta_a/b | 1〜5 | 1未満は重いテールでロスが増えやすい |
| rework_reinject_ratio | 0.2〜1.0 | 1.0は再投入フル。0.1未満は影響薄 |
| conditional_prob_ratio | 0.4〜0.9 | 0.95超はCONDばかりで循環しやすい |

### 承認者構成
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| n_senior | 0〜5 | 多いほどDR容量/品質が上がる |
| n_coordinator | 0〜6 | 中程度の影響 |
| n_new | 0〜10 | 多すぎると品質低下側に寄る |

### 並列サーバ数と摩擦
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| n_servers_mid | 1〜10 | 1未満不可。多すぎると摩擦で逆効果も |
| n_servers_fin | 1〜10 | 同上 |
| friction_model | linear / pairs | pairs は摩擦が強く出る |
| friction_alpha | 0.02〜0.08 (linear) / 0.01〜0.03 (pairs) | 0.1超で遅延が大きくなりやすい |

### Scheduler / LatentRisk
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| engineer_pool_size | 4〜30 | 少なすぎるとDR2爆発、増やしても速度は直接変わらない |
| hours_per_day_per_engineer | 2〜8 | 10超はLatentRiskが下限に張り付く |

---

## 2) 破綻しやすい組合せ（警戒パターン）
- arrival_rate 高い × bundle_size 大きい × dr_capacity 低い → **DR詰まりが支配**
- dr_period 長い × decision_latency 長い → **DR待ちがリードタイム支配**
- rework_load_factor 高い × conditional_prob_ratio 高い × decay 高め → **再作業が収束しない**
- n_servers 多い × friction_alpha 高い → **摩擦で処理時間が逆に増える**

---

## 3) 簡易キャパ判定（ざっくり目安）
以下の不等式を大きく超えるとWIPが増えやすい。

- **DR1の詰まり判定**
  - `arrival_rate / bundle_size_small`  
    `<= dr1_capacity / dr1_period`

- **DR2の詰まり判定**
  - `arrival_rate / (bundle_size_small * bundle_size_mid)`  
    `<= dr2_capacity / dr2_period`

- **DR3の詰まり判定**
  - `arrival_rate / (bundle_size_small * bundle_size_mid * bundle_size_fin)`  
    `<= dr3_capacity / dr3_period`

- **作業ゲートの詰まり判定**
  - `arrival_rate / bundle_size_small <= n_servers_mid / mid_exp_duration`
  - `arrival_rate / (bundle_size_small * bundle_size_mid) <= n_servers_fin / fin_exp_duration`

※ 差し戻しがあるので、実際の負荷は上記より増えます。

---

## 4) 破綻サイン（出力からの早期検知）
- `scenario_scorecard.png` で LT(P90) が急伸、TP が極小
- `wip_time_series.png` で WIPが右肩上がり
- `gate_wait_heatmap.png` で特定ゲートが極端に赤い
- `loss_breakdown.csv` で TimeLoss がベースの数倍

---

## 5) 使い方の勘所
- まず **arrival_rate / bundle_size** と **DR capacity** の釣り合いを取る
- DR待ちが支配なら **dr_period短縮 or capacity増**
- Mid/Fin滞留なら **n_servers増** or **duration短縮**
- 再作業が支配なら **dr_quality / conditional_prob_ratio / rework_load_factor** を優先調整

