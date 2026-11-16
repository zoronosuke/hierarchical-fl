# Jetson 7台による flat FL（多段集約なし）実験と HFL との通信量比較

## 1. 目的

[5段HFL実験](jetson-5tier-experiment.md)・[10分間隔の長時間実験](jetson-longrun-experiment.md)と**同じ 7 台・同じ親子リンク・同じ 6 partition・同じモデルと学習設定**のまま、集約を Global だけで行う通常の連合学習（flat FL）を動かし、**多段集約（HFL）のほうが通信量が少ない**ことを実測で示す。

各クライアントは Global に直接つなぐのではなく、HFL と同じ親子リンクをたどる。中継ノード（Edge 01/02、L01、L03）は **子の Flower gRPC 接続を中身を読まずに親へ TCP 転送するだけ**で、集約しない。中継ノード自身は Internal Client として学習に参加する（HFL と同じ client_id・partition）。

```
                 Global (:8080)  ← 集約はここだけ。6 クライアントが Flower で直接つながる
             ┌──────────┴──────────┐
       Edge 01 (:9001)        Edge 02 (:9002)    ← 中継 (TCP 転送) + Internal Client
     ┌─────┼──────┐               │
   L01   L02   Internal        Internal
  (:9003)                                        ← 中継 (TCP 転送) + Internal Client
 ┌──┴──┐
L03   Internal                                   ← 中継 (TCP 転送) + Internal Client
(:9004)
 ┌┴──┐
L04  Internal
```

### HFL と何が同じで何が違うか

| 項目 | HFL (`jetson-5tier-*`) | flat FL (`jetson-flat*`) |
|---|---|---|
| Global | Flower サーバー (FedAvg、間隔・チェックポイント・停止ファイル) | 同じ。`min_available_clients` だけ 2 → 6 |
| クライアント | `HFLClient` (Flower gRPC) | 同じクラス・同じ 6 client_id・同じ partition |
| 中継ノードの中身 | Flower sub-server で子を集約 → 自分が Flower クライアントとして親へ 1 モデル | 子の接続を TCP 転送 (`src/core/tcp_relay.py`)。集約なし |
| 中継の学習 | Internal Client が自分の sub-server に接続 | Internal Client が親へ直接接続 |
| 1 モデルの大きさ・通信方式 | 0.755 MB、Flower gRPC | 同じ (TCP の中身をそのまま通すため) |
| リンクを通るモデル数 (片方向/ラウンド) | 全リンク 1 | 子の部分木のクライアント数 |

通信方式・モデルサイズ・学習条件がそろうので、差は「どこで集約するか = リンクを通るモデルの数」だけになる。

### 期待される通信量（1 ラウンド・片方向のモデル転送数）

| リンク (子 → 親) | HFL | flat FL |
|---|---:|---:|
| Edge 01 → Global | 1 | 5 (edge_01_internal, leaf_01_internal, leaf_02, leaf_03_internal, leaf_04) |
| Edge 02 → Global | 1 | 1 |
| L01 → Edge 01 | 1 | 3 |
| L02 → Edge 01 | 1 | 1 |
| L03 → L01 | 1 | 2 |
| L04 → L03 | 1 | 1 |
| **合計** | **6** | **13** |

HFL では 1 モデル = 0.755 MB の実測から、flat FL は全リンク合計で約 2.2 倍（Edge 01–Global は 5 倍、L01–Edge 01 は 3 倍）の通信量になる見込み。

## 2. プロファイル

| プロファイル | ラウンド間隔 | 用途 |
|---|---|---|
| `config/jetson-flat/` | **なし（連続実行）**。長さは `--max-hours` で決める (既定 48 時間) | 既定。HFL 側の対になる連続実行プロファイルは `jetson-5tier-longrun` (通信量の記録 `sampler_interval_s` を持つ) |
| `config/jetson-flat-paced/` | 600 秒・時計合わせ (HFL の `jetson-5tier-paced` と同じ) | 時間軸でも HFL の paced 実験と比べたいとき |
| `config/local-flat/` | なし | 1PC での疎通確認 (`run.py`) |

両プロファイルの中身:

| ファイル | 内容 |
|---|---|
| `nodes.yaml` | 台帳。番号・ポート・親は HFL と同じ。role は 202/203/204/206 が `relay`、205/207 が `leaf` |
| `global.yaml` | `min_fit_clients: 6`、`min_available_clients: 6`、`round_timeout: 1500`、`checkpoint_every: 1` |
| `topology.yaml` | `mode: flat`、`relays:` (edges の代わり。`config_file` と `leaf_clients`)、`data_partition` は HFL と同一 |
| `edge_01.yaml` ほか 4 つ | 中継の設定: `listen_address`、`parent_address`、`startup_timeout`、`shutdown_timeout`、`internal_client.enabled` |

中継の設定は間隔に依存するタイムアウト検証を持たないので、どちらのプロファイルでも同じ内容。

| Jetson | IP | 役割 | 待受 → 接続先 | Partition |
|---|---|---|---|---:|
| `.201` | `192.168.10.201` | Global (集約) | `0.0.0.0:8080` | — |
| `.202` | `192.168.10.202` | edge_01 中継 + Internal | `:9001` → `.201:8080` | 0 |
| `.203` | `192.168.10.203` | edge_02 中継 + Internal | `:9002` → `.201:8080` | 3 |
| `.204` | `192.168.10.204` | leaf_01 中継 + Internal | `:9003` → `.202:9001` | 1 |
| `.205` | `192.168.10.205` | leaf_02 Leaf | → `.202:9001` | 2 |
| `.206` | `192.168.10.206` | leaf_03 中継 + Internal | `:9004` → `.204:9003` | 4 |
| `.207` | `192.168.10.207` | leaf_04 Leaf | → `.206:9004` | 5 |

UFW・CIFAR-10 キャッシュ・SSH 鍵・時計の準備は HFL と同じ（待受ポートも同じ）。

## 3. 起動・監視・停止・回収

HFL と同じ `scripts/hfl.sh`。**`--profile` を flat のものにする**だけ。

```bash
cd ~/hierarchical-fl-persistent
export HFL_PROFILE=config/jetson-flat          # 10 分間隔なら config/jetson-flat-paced
scripts/hfl.sh check                            # 台帳と設定の整合性 (relay の listen/parent アドレス)
scripts/hfl.sh start-all --run-id flat-YYYYMMDD-01-dry --dry-run   # 疎通確認 (2 ラウンド)
scripts/hfl.sh status-all --run-id flat-YYYYMMDD-01-dry           # 7 台 finished
scripts/hfl.sh stop-all   --run-id flat-YYYYMMDD-01-dry           # 記録プロセスを止める

scripts/hfl.sh start-all --run-id flat-YYYYMMDD-01 --max-hours 24  # 本番 (連続実行)
scripts/hfl.sh monitor start --run-id flat-YYYYMMDD-01             # http://192.168.10.201:8000/
scripts/hfl.sh stop-all  --run-id flat-YYYYMMDD-01                 # 終わったら (時間上限でも学習は自動停止)
scripts/hfl.sh collect   --run-id flat-YYYYMMDD-01                 # PC から。collected/<run-id>/
```

- 起動順は台帳の順 (Global → 中継 → Leaf)。中継は**親の待受を見てから** listen するので、`start-all` の `wait_listener` の順序と整合する。
- Global は 6 クライアントがそろうまで 1 ラウンド目を始めない。
- 停止は HFL と同じ: Global がラウンド境界で止まる → 切断が中継を通って各クライアントへ届く → 中継は中継中の接続が無くなるのを待って終了する (`shutdown_timeout`)。
- `resume` も同じ (Global のチェックポイント。中継は状態を持たない)。
- 監視画面では中継ノードは「待機 (子 N 接続を転送中)」のように表示され、Internal Client の学習中は青になる。線の横の通信量は HFL と同じ尺度 (子の `links.parent`)。

## 4. 成功判定

1. Global の events で毎ラウンドの `aggregate` が `results=6`、`participants` が 6 クライアント (`edge_01_internal`, `edge_02_internal`, `leaf_01_internal`, `leaf_02`, `leaf_03_internal`, `leaf_04`)。
2. 全ノードの `hfl-<NUM>.exit` が `0`。events の `stop` の `reason`: Global は `max_duration` / `stop_file`、中継は `finished` (`relay`) と `finished` (`internal_client`)、Leaf は `finished`。
3. 中継の events に `relay_connect` / `relay_disconnect` があり、`relay_connect_failed` が無い。`bytes_up > 0` の `relay_disconnect` (= Flower の接続。子の `wait_for_port` の疎通確認は上り 0 バイトの短い接続として別に記録される) の数: edge_01 = 4、leaf_01 = 2、leaf_03 = 1、edge_02 = 0。`stop` に転送した合計 (`connections`, `bytes_up`, `bytes_down`) が付く。
4. `error` イベントとログの `Traceback` が無い。終了後、全台の `pgrep -af '[s]rc.core'` に残りが無い。

## 5. HFL との比較

```bash
uv run python scripts/compare_runs.py collected/paced-20261014-01 collected/flat-YYYYMMDD-01 \
    --labels HFL FlatFL                      # → compare/paced-20261014-01_vs_flat-YYYYMMDD-01/
```

監視役のフォルダ (`logs/<run-id>/monitor`) を渡すときは `--profile-a config/jetson-5tier-paced --profile-b config/jetson-flat` で台帳を指定する。

| 出力 | 内容 |
|---|---|
| `comparison.json` | リンクごと (上り・下り MB/ラウンド、期待モデル数、B/A 比)、ノードごと (送信・受信・学習分・その他)、全体 (MB/ラウンド、10 分間隔 1 日 144 ラウンド換算 GB、比)、精度 (最終・最高・閾値到達ラウンド、ラウンド所要時間、間隔) |
| `links.csv`, `nodes.csv` | 表計算用 |
| `link_bytes_per_round.png` | リンク × {HFL, flat} の上り・下り MB/ラウンド。棒の上の ×N = 期待モデル数 |
| `node_tx_per_round.png` | ノードごとの送信 MB/ラウンド (学習分とその他の積み上げ) |
| `cumulative_tx_vs_round.png` | 全ノード合計の学習送信量 (累計) vs ラウンド |
| `accuracy_vs_round.png` / `accuracy_vs_time.png` | Global accuracy の重ね描き |

比べ方の要点:

- リンクの量は**子ノードの `links.parent`** (`sent` = 上り、`recv` = 下り) の累計をラウンド数で割る。学習の接続の中身だけを数えるので、**ラウンド間隔 (10 分待つ / 連続) に左右されない**。SSH 監視などの「その他」の送信は時間に比例するため主図には使わない。
- 比較の基本軸は**ラウンド**。時間軸の図は補助で、両 run の間隔を注記する。
- 期待値 (1 ラウンド片方向のモデル数) は、HFL は全リンク 1、flat は子の部分木で fit を記録したクライアント数から自動で求める。

## 6. 1PC での確認

```bash
uv run python run.py --dry-run --global-config config/local-flat/global.yaml \
    --topology-config config/local-flat/topology.yaml --run-id local-flat
```

`tests/test_integration.py::test_flat_relay_two_global_rounds` がこれを自動化している (6 クライアントの fit × 2、Global の `aggregate results=6`、中継 4 本の正常終了と転送バイト数)。中継本体は `tests/test_tcp_relay.py`、台帳は `tests/test_hfl_nodes.py`、比較は `tests/test_compare_runs.py`。
