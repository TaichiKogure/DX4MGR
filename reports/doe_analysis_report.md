# DOE感度分析サマリーレポート

## 1. 分析概要
- 試行回数: 100回
- 目的: 各種パラメータがプロジェクトの成果（Gain）、完了数、コストに与える影響を特定する。

## 2. パラメータ感度ランキング (相関係数ベース)

### ターゲット指標: total_gain
- **cost_per_review**: 感度=中 (0.235), 方向=負
- **dr1_period**: 感度=低 (0.146), 方向=正
- **uncertainty_threshold**: 感度=低 (0.049), 方向=負
- **rework_load_factor**: 感度=低 (0.035), 方向=負
- **maturity_threshold**: 感度=低 (0.007), 方向=負
- **dr_threshold**: 感度=低 (0.003), 方向=正
- **decay**: 感度=低 (0.001), 方向=負

### ターゲット指標: completed_jobs
- **dr1_period**: 感度=低 (nan), 方向=負
- **dr_threshold**: 感度=低 (nan), 方向=負
- **cost_per_review**: 感度=低 (nan), 方向=負
- **rework_load_factor**: 感度=低 (nan), 方向=負
- **decay**: 感度=低 (nan), 方向=負
- **uncertainty_threshold**: 感度=低 (nan), 方向=負
- **maturity_threshold**: 感度=低 (nan), 方向=負

### ターゲット指標: dr_cost
- **cost_per_review**: 感度=高 (1.000), 方向=正
- **dr1_period**: 感度=低 (0.160), 方向=正
- **decay**: 感度=低 (0.150), 方向=正
- **maturity_threshold**: 感度=低 (0.138), 方向=正
- **rework_load_factor**: 感度=低 (0.082), 方向=負
- **dr_threshold**: 感度=低 (0.041), 方向=正
- **uncertainty_threshold**: 感度=低 (0.009), 方向=正

## 3. 結論

### 感度が高いパラメータ (有効なレバー)
- cost_per_review
  - これらのパラメータを調整することで、プロジェクトの成功率や成果を大きくコントロール可能です。

### 感度が低いパラメータ (あまり影響しないもの)
- 特になし
  - これらのパラメータは現状のシミュレーション範囲内では結果への影響が限定的です。