# Jetson Orin Nano 7台による3層HFL実験手順

## 1. 目的

Jetson Orin Nano 7台で、Global–Edge–Leafの3層階層型連合学習を実行します。データセットは実際のCIFAR-10、モデルはSimpleCNN、データ分割はDirichlet分布によるNon-IIDです。

Tera TermのSSH接続とPowerShellの`ssh`コマンドは、どちらもJetsonの22番ポートで動作する同じSSH接続です。本章のLinuxコマンドは、Tera Termで各Jetsonへログインして実行できます。

## 2. ネットワークとPartition

| Jetson | IP | 役割 | 待受・接続先 | Partition |
|---|---|---|---|---:|
| `tachilab-orin01` | `192.168.10.201` | Global Server | `0.0.0.0:8080` | — |
| `tachilab-orin02` | `192.168.10.202` | Edge 01 + Internal Client | `0.0.0.0:9001` → `.201:8080` | `0` |
| `tachilab-orin03` | `192.168.10.203` | Edge 02 + Internal Client | `0.0.0.0:9002` → `.201:8080` | `3` |
| `tachilab-orin04` | `192.168.10.204` | Leaf 01 | `.202:9001` | `1` |
| `tachilab-orin05` | `192.168.10.205` | Leaf 02 | `.202:9001` | `2` |
| `tachilab-orin06` | `192.168.10.206` | Leaf 03 | `.203:9002` | `4` |
| `tachilab-orin07` | `192.168.10.207` | Leaf 04 | `.203:9002` | `5` |

全体を`total_partitions: 6`で分割します。PDF「2026610実験.pdf」のLeaf 03/04に記載されたpartition `3`/`4`は現行実装には使用しません。`3`はEdge 02のInternal Clientが使用するため、Leaf 03/04は`4`/`5`が正しい割当です。

実機用設定は次を使用します。

- Global: `config/jetson-7node/global.yaml`
- Edge 01/02: `config/jetson-7node/edge_01.yaml`、`edge_02.yaml`
- Partition: `config/topology.yaml`
- 学習条件: `config/defaults.yaml`

既定条件はGlobal 5ラウンド、Edge sub-round 1、local epoch 2、batch size 32、SGD、学習率0.01、Dirichlet alpha 0.5、seed 42です。

## 3. 初回セットアップとCIFAR-10の準備

各Jetsonで一度だけセットアップします。CIFAR-10を取得するときはインターネットへ接続できるネットワークを使用します。

```bash
cd ~/hierarchical-fl-persistent
bash scripts/setup_jetson.sh
.venv/bin/python -c "from datasets import load_dataset; load_dataset('uoft-cs/cifar10')"
```

7台すべてでキャッシュが作成されたら、実験用ネットワークへ接続します。各JetsonでIPを確認してください。

```bash
hostname
hostname -I
```

UFWが有効な場合だけ、Globalでは8080、Edge 01では9001、Edge 02では9002への実験LAN内通信を許可します。UFW全体を無効化する必要はありません。

```bash
sudo ufw status
# Globalだけ: sudo ufw allow from 192.168.10.0/24 to any port 8080 proto tcp
# Edge 01だけ: sudo ufw allow from 192.168.10.0/24 to any port 9001 proto tcp
# Edge 02だけ: sudo ufw allow from 192.168.10.0/24 to any port 9002 proto tcp
```

## 4. 全Jetsonで行う実験前準備

Tera Termで7台へログインし、各画面で次を実行します。`RUN_ID`は7台で同じ文字列にしてください。

```bash
cd ~/hierarchical-fl-persistent

export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NO_PROXY=localhost,127.0.0.1,192.168.10.201,192.168.10.202,192.168.10.203,192.168.10.204,192.168.10.205,192.168.10.206,192.168.10.207
export no_proxy="$NO_PROXY"
export no_grpc_proxy="$NO_PROXY"

RUN_ID=manual-20260803-01
mkdir -p "logs/$RUN_ID"
pgrep -af '[s]rc.core' || true
```

`pgrep`で別のHFL実験が表示された場合は、新しい実験を開始せず、実行者とプロセスを確認します。

## 5. 7台で手動起動するコマンド

順番はGlobal → Edge 2台 → Leaf 4台です。各コマンドはフォアグラウンドで実行されるため、実験中はTera Termを閉じないでください。

### 5.1 Global Server: 192.168.10.201

```bash
set -o pipefail
.venv/bin/python -m src.core.global_server \
  --config config/jetson-7node/global.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-201.log"
```

別のTera Term画面で、8080番ポートの待受を確認してからEdgeを起動します。

```bash
ss -ltn | grep ':8080 '
```

### 5.2 Edge 01: 192.168.10.202

```bash
set -o pipefail
.venv/bin/python -m src.core.run_edge \
  --edge-config config/jetson-7node/edge_01.yaml \
  --global-config config/jetson-7node/global.yaml \
  --topology-config config/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-202.log"
```

### 5.3 Edge 02: 192.168.10.203

```bash
set -o pipefail
.venv/bin/python -m src.core.run_edge \
  --edge-config config/jetson-7node/edge_02.yaml \
  --global-config config/jetson-7node/global.yaml \
  --topology-config config/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-203.log"
```

Edgeの起動後、`.202`では9001、`.203`では9002の待受を確認してからLeafを起動します。

```bash
# 192.168.10.202で実行
ss -ltn | grep ':9001 '

# 192.168.10.203で実行
ss -ltn | grep ':9002 '
```

### 5.4 Leaf 01: 192.168.10.204

```bash
set -o pipefail
.venv/bin/python -m src.core.run_leaf \
  --client-id leaf_01 \
  --edge-address 192.168.10.202:9001 \
  --partition-id 1 \
  --global-config config/jetson-7node/global.yaml \
  --topology-config config/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-204.log"
```

### 5.5 Leaf 02: 192.168.10.205

```bash
set -o pipefail
.venv/bin/python -m src.core.run_leaf \
  --client-id leaf_02 \
  --edge-address 192.168.10.202:9001 \
  --partition-id 2 \
  --global-config config/jetson-7node/global.yaml \
  --topology-config config/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-205.log"
```

### 5.6 Leaf 03: 192.168.10.206

```bash
set -o pipefail
.venv/bin/python -m src.core.run_leaf \
  --client-id leaf_03 \
  --edge-address 192.168.10.203:9002 \
  --partition-id 4 \
  --global-config config/jetson-7node/global.yaml \
  --topology-config config/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-206.log"
```

### 5.7 Leaf 04: 192.168.10.207

```bash
set -o pipefail
.venv/bin/python -m src.core.run_leaf \
  --client-id leaf_04 \
  --edge-address 192.168.10.203:9002 \
  --partition-id 5 \
  --global-config config/jetson-7node/global.yaml \
  --topology-config config/topology.yaml \
  --defaults-config config/defaults.yaml \
  2>&1 | tee "logs/$RUN_ID/hfl-207.log"
```

## 6. 成功判定

次をすべて満たせば完了です。

1. Globalが5ラウンド完了し、`Global Server finished.`を出力する。
2. Edge 01/02が各ラウンドで`results=3`を出力する。
3. Leaf 4台とInternal Client 2個が、それぞれ5回の`fit`を完了する。
4. Globalの評価結果にroundごとのlossとaccuracyが出力される。
5. `Traceback`、`HTTP proxy returned response code 403`、timeoutがない。
6. 終了後、全Jetsonの`pgrep -af '[s]rc.core'`に今回のプロセスが残っていない。

中断するときは、まずPIDを確認し、今回のプロセスだけへ`TERM`を送ります。他の研究プロセスを巻き込む可能性があるため、確認なしで広い`pkill`を実行しないでください。

```bash
pgrep -af '[s]rc.core'
kill -TERM <確認したPID>
```

## 7. PowerShellからの自動配備・起動

手動実験と同じ構成をWindowsから起動する場合に使用します。既存の`~/hierarchical-fl`には触れず、専用の`~/hierarchical-fl-persistent`を使用します。

```powershell
cd "C:\Users\iamzo\Documents\01_School\University\Research\02_Lab\FOWERD\Dev\hierarchical-fl"

.\scripts\deploy_jetsons.ps1 `
  -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519"

.\scripts\start_jetsons.ps1 `
  -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519" `
  -DryRun

.\scripts\start_jetsons.ps1 `
  -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519"
```

起動スクリプトはGlobalとEdgeの待受確認後に次の層を起動し、ログを`logs/<RunId>/`へ保存します。

## 8. Git更新時の注意

Jetson上の既存研究ファイルを保護するため、PDF記載の`git reset --hard origin/master`は使用しません。配備スクリプトは専用ディレクトリに追跡済みのローカル変更があれば停止します。停止した場合は`git status --short`と`git diff`を確認し、変更の所有者を確認してから対処してください。

生ログ、TensorBoardログ、キャッシュはGitへ登録しません。Gitへ登録するのは設定、手順書、再利用可能なスクリプト、実験結果の要約です。

## 9. 2026-08-03 CIFAR-10実行結果

コミット`09fdb5a`、Global 5ラウンド、local epoch 2、Dirichlet alpha 0.5、seed 42で7台実験を完走しました。Global実行時間は91.31秒でした。

| Round | Loss | Accuracy |
|---:|---:|---:|
| 0 | — | 0.1002 |
| 1 | 1.9459 | 0.2755 |
| 2 | 1.4173 | 0.4532 |
| 3 | 1.2080 | 0.5496 |
| 4 | 1.0770 | 0.6026 |
| 5 | 1.0031 | 0.6320 |

Global、Edge 2台、Leaf 4台、Internal Client 2個は正常終了し、traceback、gRPC proxyエラー、timeoutはありませんでした。生ログはローカルの`artifacts/jetson-7node-cifar10-20260803/`に保存し、Gitには含めません。
