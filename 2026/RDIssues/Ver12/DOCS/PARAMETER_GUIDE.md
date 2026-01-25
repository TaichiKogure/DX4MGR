# DX4MGR Ver12 パラメーターガイド (簡易)

このファイルは概要のみです。詳細は `PARAMETER_GUIDE_Ver12.md` を参照してください。

- `bundle_size_*`: バンドルのサイズ。小さいほど待ちが減るがDR負荷は増える
- `dr*_period`: DR会議の間隔。短くすると待ちが減る
- `dr*_capacity`: DR会議で処理できる件数
- `n_servers_mid` / `n_servers_fin`: Mid/Finの並列数
- `n_senior` / `n_coordinator` / `n_new`: 承認者の構成 (品質と容量に影響)
