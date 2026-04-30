# Phase 3 利用マニュアル

## 概要
Phase 3 では、Phase 2 で生成した 2D 電極構造を入力として、電解液相におけるイオン輸送を 2 次元でシミュレーションする機能が追加されました。
局所的な濃度偏り、ポテンシャル分布、見かけ輸送抵抗、厚み方向の濃度分布を計算・可視化できます。

## 実行方法

### 1. CLI での実行
単一のシミュレーションを実行する場合：
```bash
python -m iondynamics.cli transport2d run --config configs/default.yaml
```

複数ケースの比較（厚み違いなど）を並列実行する場合：
```bash
python -m iondynamics.cli transport2d compare --config configs/default.yaml --axis thickness --values 60 80 100
```

### 2. GUI での実行
1. `run_gui.bat` で GUI を起動します。
2. 「Transport 2D」タブを選択します。
3. **Microstructure Source** で構造を再利用するか、新規生成するかを選択します。
   - `Use generated Microstructure`: 「Microstructure」タブですでに生成済みの構造がある場合、それを再利用します。
   - `Regenerate before run`: 実行のたびに新しい構造を生成します。
4. 時間刻み (Time Step)、終了時間 (Final Time)、初期濃度、Cレートなどのパラメータを設定します。
5. 「Run 2D Transport」ボタンを押すと計算が開始されます。
6. 完了後、右側のプレビューエリアで濃度分布 (Concentration)、電位分布 (Potential)、厚み方向プロファイル (Profiles) を切り替えて確認できます。
7. 「Open Output Folder」ボタンから、詳細な結果が保存されたフォルダを直接開くことができます。

## 出力内容
実行結果は `outputs/runs/` または指定した出力ディレクトリに保存されます。

- `ce_final_2d.png`: 最終時刻の 2D 濃度分布ヒートマップ。
- `ce_final_2d_with_microstructure.png`: 濃度分布に活物質の輪郭を重ねた図。
- `phi_e_steady.png`: 定常状態の電解液ポテンシャル分布。
- `phi_e_steady_with_microstructure.png`: ポテンシャル分布に活物質の輪郭を重ねた図。
- `j_norm_steady.png`: 電流密度分布。
- `ce_profiles_1d.png`: 厚み方向平均濃度の時間推移プロファイル。
- `phase_map_used.png`: シミュレーションに使用された相マップ。
- `interface_mask.png`: ソース項（反応）が適用された界面の位置を示すマスク画像。
- `transport_kpis.csv`: 見かけ輸送抵抗、局所枯渇率などの主要 KPI。
- `transport_fields.npz`: 濃度・電位・電流密度の全数値データ。
- `solver_config.yaml`: 実行時に使用されたソルバの設定。

## 主要なパラメータ
- **Time Step (s)**: 計算の時間刻み。小さいほど精度は上がりますが計算時間がかかります。
- **Final Time (s)**: シミュレーションを終了する時間。放電末期を見るには十分な長さを設定してください。
- **BC Separator**: セパレータ側の境界条件。「constant_concentration（一定濃度）」または「constant_flux（一定流束）」を選択可能です。
- **C-rate**: 電流密度を決定します。

## 注意事項
- 2D 輸送計算は 1D モデルに比べて計算負荷が高いため、解像度や時間刻みの設定に注意してください。
- 現在のモデルは電解液相のみを対象としており、活物質内部の拡散は考慮されていません（界面での反応ソースとして扱われます）。
