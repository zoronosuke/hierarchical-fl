# 設計決定記録

## パッケージ管理: uv を採用

### 理由
- uv は aarch64 (ARM64) の wheel を公式提供しており Jetson で動作する
- `[[tool.uv.index]]` + `[tool.uv.sources]` で Jetson 用 PyTorch インデックスを
  `platform_machine == 'aarch64'` のマーカーで自動切り替え可能
- pip + venv より10倍以上高速

### Jetson 固有の注意
- JetPack のシステム Python (3.10) に CUDA バインディングが pre-installed
- `pypi.jetson-ai-lab.io/jp6/cu126` から PyTorch を取得する必要がある
- uv の venv からシステムの CUDA ライブラリを参照する設定が必要になる可能性あり
  → 問題が出たら `--system-site-packages` 相当の設定を検討

## 階層型FL: Flower 入れ子構造を採用

### 理由
- Flower の gRPC 通信を全階層で利用可能
- 独自プロトコル不要
- Edge の fit() 内で start_server() を呼ぶだけ

### 制約
- start_server / start_client は deprecated (1.13+)
- 新しい SuperLink/SuperNode API は入れ子構造に非対応
- Flower のメジャーバージョンアップ時に対応が必要

## 同期制御: 同期型 + タイムアウト

### 理由
- 実装がシンプル
- Edge 2-3台, Leaf 5-10台の規模では非同期の利点が薄い
- タイムアウトで障害時の進行を保証

## 集約戦略: FedAvg + Strategy Registry

### 理由
- FedAvg がベースライン
- @register_strategy デコレータで将来の戦略追加が容易
- contribution_factor で Edge/Leaf の寄与度を設定ファイルから調整可能
