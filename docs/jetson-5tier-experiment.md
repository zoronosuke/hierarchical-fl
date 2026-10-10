# Jetson Orin Nano 7台による5段HFL実験手順

## 1. 目的

[3層HFL実験](jetson-7node-experiment.md)と同じJetson 7台・CIFAR-10・SimpleCNN・Dirichlet Non-IIDのまま、トポロジーを5段に変えて実行します。初回セットアップ、CIFAR-10の準備、実験前の環境変数設定、成功判定の考え方、中断方法は3層実験の手順書と同じです。本書は差分だけを記載します。

```
                 Global Server (:8080)            ← 1段目
             ┌──────────┴──────────┐
       Edge 01 (:9001)        Edge 02 (:9002)     ← 2段目
     ┌─────┼──────┐               │
   L01   L02   Internal        Internal
  (:9003)                                         ← 3段目
 ┌──┴──┐
L03   Internal                                    ← 4段目
(:9004)
 ┌┴──┐
L04  Internal                                     ← 5段目
```

L01とL03は**中継ノード**です。上位サーバーのクライアントとして参加しつつ、自身も子サーバーを立てて下位を集約します。実装はEdgeと同じ`src.core.run_edge`で、各中継ノードは自身のInternal Clientでも学習します。

## 2. ネットワークとPartition

| Jetson | IP | 役割 | 待受・接続先 | Partition |
|---|---|---|---|---:|
| `tachilab-orin01` | `192.168.10.201` | Global Server | `0.0.0.0:8080` | — |
| `tachilab-orin02` | `192.168.10.202` | Edge 01 + Internal | `0.0.0.0:9001` → `.201:8080` | `0` |
| `tachilab-orin03` | `192.168.10.203` | Edge 02 + Internal | `0.0.0.0:9002` → `.201:8080` | `3` |
| `tachilab-orin04` | `192.168.10.204` | L01 中継 + Internal | `0.0.0.0:9003` → `.202:9001` | `1` |
| `tachilab-orin05` | `192.168.10.205` | L02 Leaf | `.202:9001` | `2` |
| `tachilab-orin06` | `192.168.10.206` | L03 中継 + Internal | `0.0.0.0:9004` → `.204:9003` | `4` |
| `tachilab-orin07` | `192.168.10.207` | L04 Leaf | `.206:9004` | `5` |

各Jetsonが使うPartitionは3層実験と同じです。違いは集約経路だけなので、3層実験の結果と直接比較できます。

設定ファイルはすべて`config/jetson-5tier/`にあります。

| ファイル | 内容 |
|---|---|
| `global.yaml` | 3層実験と同じGlobal設定（5ラウンド、round_timeout 4200秒） |
| `topology.yaml` | 親子関係とPartition |
| `edge_01.yaml` / `edge_02.yaml` | Edge 01（参加者3）、Edge 02（参加者1: Internalのみ） |
| `leaf_01.yaml` / `leaf_03.yaml` | 中継ノード（参加者2: 子1台 + Internal） |

### タイムアウトの入れ子

下位ほど短くし、各ノードの`parent_result_timeout`が直上サーバーのラウンドタイムアウトより短くなるようにしています。起動時に検証され、条件を満たさないと`ValueError`で停止します。

| ノード | sub_round_timeout | parent_result_timeout | 直上のround_timeout |
|---|---:|---:|---:|
| Edge 01 | 3000 | 3300 | 4200 (Global) |
| Edge 02 | 3600 | 3900 | 4200 (Global) |
| L01 | 2400 | 2700 | 3000 (Edge 01) |
| L03 | 1800 | 2100 | 2400 (L01) |

中継ノードの設定にある`parent_round_timeout`は直上Edgeの`sub_round_timeout`と同じ値にします。`sub_rounds`を2以上に変える場合は、子孫の`upstream_sub_rounds`を祖先Edgeの`sub_rounds`の積に変更してください。

## 3. ファイアウォール

UFWが有効な場合だけ、各待受ポートを許可します。

```bash
# .201: sudo ufw allow from 192.168.10.0/24 to any port 8080 proto tcp
# .202: sudo ufw allow from 192.168.10.0/24 to any port 9001 proto tcp
# .203: sudo ufw allow from 192.168.10.0/24 to any port 9002 proto tcp
# .204: sudo ufw allow from 192.168.10.0/24 to any port 9003 proto tcp
# .206: sudo ufw allow from 192.168.10.0/24 to any port 9004 proto tcp
```

## 4. 実験前準備

3層実験手順書の「4. 全Jetsonで行う実験前準備」を7台で実行します。`RUN_ID`は別の名前にしてください（例: `manual-5tier-20261010-01`）。

## 5. 7台で手動起動するコマンド

**順番が重要です。** 親の待受を`ss -ltn`で確認してから子を起動します。

Global → Edge 01/02 → L01 → L02 → L03 → L04

### 5.1 Global Server: 192.168.10.201

```bash
set -o pipefail
.venv/bin/python -m src.core.global_server \
  --config config/jetson-5tier/global.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-201.log"
```

`ss -ltn | grep ':8080 '`で待受を確認します。

### 5.2 Edge 01: 192.168.10.202

```bash
set -o pipefail
.venv/bin/python -m src.core.run_edge \
  --edge-config config/jetson-5tier/edge_01.yaml \
  --global-config config/jetson-5tier/global.yaml \
  --topology-config config/jetson-5tier/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-202.log"
```

### 5.3 Edge 02: 192.168.10.203

```bash
set -o pipefail
.venv/bin/python -m src.core.run_edge \
  --edge-config config/jetson-5tier/edge_02.yaml \
  --global-config config/jetson-5tier/global.yaml \
  --topology-config config/jetson-5tier/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-203.log"
```

`.202`で`ss -ltn | grep ':9001 '`を実行し、待受を確認してから次へ進みます。

### 5.4 L01（中継）: 192.168.10.204

```bash
set -o pipefail
.venv/bin/python -m src.core.run_edge \
  --edge-config config/jetson-5tier/leaf_01.yaml \
  --global-config config/jetson-5tier/global.yaml \
  --topology-config config/jetson-5tier/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-204.log"
```

### 5.5 L02（Leaf）: 192.168.10.205

```bash
set -o pipefail
.venv/bin/python -m src.core.run_leaf \
  --client-id leaf_02 \
  --edge-address 192.168.10.202:9001 \
  --partition-id 2 \
  --global-config config/jetson-5tier/global.yaml \
  --topology-config config/jetson-5tier/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-205.log"
```

`.204`で`ss -ltn | grep ':9003 '`を実行し、待受を確認してから次へ進みます。

### 5.6 L03（中継）: 192.168.10.206

```bash
set -o pipefail
.venv/bin/python -m src.core.run_edge \
  --edge-config config/jetson-5tier/leaf_03.yaml \
  --global-config config/jetson-5tier/global.yaml \
  --topology-config config/jetson-5tier/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-206.log"
```

`.206`で`ss -ltn | grep ':9004 '`を実行し、待受を確認してから次へ進みます。

### 5.7 L04（Leaf）: 192.168.10.207

```bash
set -o pipefail
.venv/bin/python -m src.core.run_leaf \
  --client-id leaf_04 \
  --edge-address 192.168.10.206:9004 \
  --partition-id 5 \
  --global-config config/jetson-5tier/global.yaml \
  --topology-config config/jetson-5tier/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-207.log"
```

## 6. 成功判定

次をすべて満たせば完了です。

1. Globalが5ラウンド完了し、`Global Server finished.`を出力する。
2. 各ラウンドの`Parent generation complete`で、`results`が次の値になる。
   - Edge 01: `results=3`
   - Edge 02: `results=1`
   - L01: `results=2`
   - L03: `results=2`
3. `edge_01_internal`、`leaf_01_internal`、`leaf_02`、`edge_02_internal`、`leaf_03_internal`、`leaf_04`が、それぞれ5回の`fit`を完了する。
4. Globalの評価結果に、ラウンドごとのlossとaccuracyが出力される。
5. `Traceback`、`HTTP proxy returned response code 403`、timeoutがない。
6. 終了後、全Jetsonの`pgrep -af '[s]rc.core'`に今回のプロセスが残っていない。

## 7. PowerShellからの自動配備・起動

配備は3層実験と同じ`deploy_jetsons.ps1`を使います。起動には5段用の`start_jetsons_5tier.ps1`を使います。このスクリプトは各中継ノードの待受を確認してから、その子を起動します。

```powershell
.\scripts\deploy_jetsons.ps1 -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519"

.\scripts\start_jetsons_5tier.ps1 -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519" -DryRun
.\scripts\start_jetsons_5tier.ps1 -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519"
```

## 8. 1PCでの疎通確認

実機の前に、1台のPCで5段構成のdry-runを実行できます（設定: `config/local-5tier/`）。

```bash
uv run python run.py --dry-run --topology-config config/local-5tier/topology.yaml
```

`tests/test_integration.py::test_persistent_hfl_5tier_two_global_rounds`は、このdry-runを2ラウンド実行して検証するテストです。
