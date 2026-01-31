# Ver14 パラメータ勘所表（破綻しないスケール目安）

## 前提
- 単位は「日」。
- 破綻 = ほぼ未完了・WIP爆増・待ち時間が支配的・DR待ちが無限に伸びる状態。
- 目安レンジは **Baseline (scenarios.csv)** を中心にした実務的な安全域。
- Ver14は **AdoptionGate**（Small後の採用判定）と **DR固定日 + 延期** を持つ。
- Small工程も `n_servers_small` で制約される（Ver13は実質無制限）。
- 摩擦は **「稼働中のサーバー数」** に基づく（Ver13の最大サーバー数ベースとは異なる）。

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

### DR固定日・周期・容量・品質
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| dr1_t / dr2_t / dr3_t | 60〜900（days内で昇順） | days外や逆順はDRが開かずTP=0に近づく |
| dr1/2/3_period | 30〜120 | 固定日が無効/未設定時の保険。180超はDR待ちが支配的 |
| dr1/2/3_capacity | 5〜30 | arrivalに比べて低いとDR詰まり |
| dr_quality | 0.4〜0.9 | 1.0近いと速度低下で詰まりやすい |
| dr_quality_speed_alpha | 0.5〜2.0 | 3超で容量補正が極端になりやすい |
| decision_latency_days | 0〜14 | 30超はリードタイム支配要因になる |

補足:
- Ver14のDRは **固定日 (dr*_t)** を優先。現行アダプタは各DR1回のみの固定日を持つ。
- 会議時点でキューが空だと **延期 (Postpone: 90 or 180日)** が発生する（パラメータでは未露出）。

### 採用ゲート（AdoptionGate）
| パラメータ | 目安レンジ | コメント / 破綻サイン |
| --- | --- | --- |
| adoption_rate | 0.4〜0.9 | 低すぎるとTPが極端に落ちる（Reject増） |

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
| n_servers_small | 2〜12 | 低すぎるとSmall待ちが発生（Ver14新規） |
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
- dr*_t が **days外** or **極端に遅い** × adoption_rate低い → **TPがほぼゼロ**
- dr*_t が少なく **会議回数が1回** × arrival_rate高い → **DR待ちが支配**
- dr_period 長い × decision_latency 長い → **DR待ちがリードタイム支配**
- n_servers_small 低い × arrival_rate 高い → **上流Small詰まり**
- rework_load_factor 高い × conditional_prob_ratio 高い × decay 高め → **再作業が収束しない**
- n_servers 多い × friction_alpha 高い → **摩擦で処理時間が逆に増える**

---

## 3) 簡易キャパ判定（ざっくり目安）
以下の不等式を大きく超えるとWIPが増えやすい。

- **Small作業の詰まり判定**
  - `arrival_rate <= n_servers_small / small_exp_duration`

- **DR1の詰まり判定（採用後流量で見る）**
  - `arrival_rate * adoption_rate / bundle_size_small`
    `<= dr1_capacity / dr1_period`

- **DR2の詰まり判定**
  - `arrival_rate * adoption_rate / (bundle_size_small * bundle_size_mid)`
    `<= dr2_capacity / dr2_period`

- **DR3の詰まり判定**
  - `arrival_rate * adoption_rate / (bundle_size_small * bundle_size_mid * bundle_size_fin)`
    `<= dr3_capacity / dr3_period`

- **作業ゲートの詰まり判定**
  - `arrival_rate <= n_servers_small / small_exp_duration`
  - `arrival_rate * adoption_rate / bundle_size_small <= n_servers_mid / mid_exp_duration`
  - `arrival_rate * adoption_rate / (bundle_size_small * bundle_size_mid) <= n_servers_fin / fin_exp_duration`

※ 差し戻しがあるので、実際の負荷は上記より増えます。
※ dr*_t が days外だと、上記判定を満たしていてもDRが起きずTPが出ません。

---

## 4) 破綻サイン（出力からの早期検知）
- `scenario_scorecard.png` で LT(P90) が急伸、TP が極小
- `wip_time_series.png` で WIPが右肩上がり
- `gate_wait_heatmap.png` で特定ゲートが極端に赤い
- `loss_breakdown.csv` で **TimeLoss / CostLoss** がベースの数倍
- `dr_gate_cycle_times.csv` で DR1/2/3(P90) が極端に長い
- `dr_gate_cycle_p90_heatmap.png` で DR突破所要(P90)が広範囲に赤い

---

## 5) 使い方の勘所
- まず **arrival_rate / bundle_size** と **DR capacity** の釣り合いを取る
- **dr*_t が days内かつ昇順** になっていることを最初に確認（TPが出ない最大要因）
- DR待ちが支配なら **dr*_t前倒し / dr_period短縮 / capacity増**
- Small滞留なら **n_servers_small増** or **small_exp_duration短縮**
- Mid/Fin滞留なら **n_servers増** or **duration短縮**
- 再作業が支配なら **dr_quality / conditional_prob_ratio / rework_load_factor** を優先調整
- adoption_rate を下げすぎると、改善が「Reject増による見かけ」になりやすい
