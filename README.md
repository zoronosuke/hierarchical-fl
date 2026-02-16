# Hierarchical Federated Learning System

3層階層型連合学習（Cloud - Edge - Client）の実装。

```
              Global Server (:8080)
              FedAvg 集約 + 評価
             ┌────────┴────────┐
         Edge 01 (:9001)    Edge 02 (:9002)
         ┌──┴──┐            ┌──┴──┐
       L01  L02  Internal  L03  L04  Internal
```

## セットアップ

### 前提条件
- Python 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 開発PC

```bash
# uv インストール (未導入の場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係インストール
uv sync

# 疎通確認
uv run python run.py --dry-run
```

### Jetson Orin Nano

```bash
bash scripts/setup_jetson.sh
uv run python run.py --dry-run
```

## 使い方

```bash
# 1PC上で全プロセス起動 (PoC)
uv run python run.py --dry-run     # 疎通確認
uv run python run.py               # 本番学習

# 個別起動 (マルチマシン)
uv run python -m src.core.global_server
uv run python -m src.core.run_edge --edge-config config/edge/edge_01.yaml
uv run python -m src.core.run_leaf --client-id leaf_01 --edge-address 127.0.0.1:9001 --partition-id 1
```

## 開発

```bash
uv run ruff check src/        # lint
uv run pytest tests/ -v        # テスト
```

## ディレクトリ構成

```
├── CLAUDE.md                  # Claude Code 用プロジェクト説明
├── .claude/skills/            # Claude Code スキル定義
├── .github/workflows/ci.yaml  # GitHub Actions CI
├── config/                    # YAML 設定ファイル群
├── src/
│   ├── core/
│   │   ├── edge_node.py       # ★ 入れ子サーバー (核心部分)
│   │   ├── client.py          # Flower Client (Leaf/Internal共用)
│   │   ├── global_server.py   # Global Server エントリポイント
│   │   └── training.py        # 学習・評価ロジック
│   ├── data/loader.py         # データセット分割 (Dirichlet)
│   ├── models/nets.py         # CNN モデル定義
│   └── strategies/            # 集約戦略 (Strategy Registry)
├── run.py                     # 一括起動
├── pyproject.toml             # uv + プロジェクト定義
└── tests/                     # テスト
```

## 技術的な注意

- `flwr.server.start_server` は deprecated だが入れ子構造では必須
- Jetson の PyTorch は `pypi.jetson-ai-lab.io` の wheel を使う (`pyproject.toml` で設定済み)
- `config/topology.yaml` でトポロジーとデータ分割を定義
