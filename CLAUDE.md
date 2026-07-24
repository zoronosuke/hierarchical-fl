# Hierarchical Federated Learning (HFL) System

## What This Project Does
3層階層型連合学習システム (Cloud - Edge - Client)。
Flower (flwr) + PyTorch で、Edge Node が Global Server のクライアントと Leaf Client のサーバーを同時に担う入れ子構造。

## Tech Stack
- Python 3.10 (Jetson JetPack 6.x 互換)
- Flower (flwr) 1.13+ — レガシー API (`start_server`/`start_client`) を意図的に使用
- PyTorch — Jetson では `pypi.jetson-ai-lab.io` の wheel を使用
- uv — パッケージ管理

## Project Structure
```
src/core/          # Flower Client/Server の実装 (edge_node.py が最重要)
src/models/        # PyTorch モデル定義
src/data/          # データセット読み込み・分割 (flwr-datasets)
src/strategies/    # 集約戦略 (FedAvg + Strategy Registry)
src/utils/         # 設定読み込み, ロギング
config/            # YAML 設定ファイル群
```

## Commands
```bash
uv sync                                    # 依存関係インストール
uv run python run.py --dry-run             # 疎通確認 (1PC上で全プロセス)
uv run python run.py                       # 本番学習
uv run pytest tests/ -v                    # テスト実行
uv run ruff check src/                     # lint
```

## Architecture (IMPORTANT)
- Edge起動時に常駐sub-serverとInternal Clientを`spawn`子プロセスで1回だけ起動
- Global Clientとsub-serverはQueueで親モデル・子集約結果を世代番号付きで同期
- sub-serverは`global_rounds * sub_rounds`を連続実行しLeaf接続を維持
- `start_server` / `start_client` は deprecated だが、入れ子構造では新 API が未対応のため使用
- `uv.lock`の検証済みFlower版とlegacy APIを使用する

## Conventions
- 設定はすべて YAML (config/ 配下)。ハードコーディング禁止
- 集約戦略は `@register_strategy` デコレータで登録、設定ファイルで切り替え
- 日本語コメントOK (研究用コードのため)
- 型ヒントは `from __future__ import annotations` を使用

## Known Issues / Gotchas
- Flower の新しい ServerApp/ClientApp API は現在の入れ子構造に移行していない
- Jetson の PyTorch は `pypi.jetson-ai-lab.io` から入れないと GPU が使えない
- `run.py` の `time.sleep()` はPoC用。本番ではヘルスチェックに置き換える
- Edge の `_estimate_total_examples()` は推定値。要改善

## Verification
変更後は必ず以下を実行:
1. `uv run ruff check src/` — lint エラーがないこと
2. `uv run pytest tests/ -v` — テストが通ること
3. `uv run python -c "import src.core.edge_node"` — import が通ること
