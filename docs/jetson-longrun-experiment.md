# Jetson 7台による長時間5段HFL実験手順（バッテリー・ソーラー運用）

## 1. 目的

[5段HFL実験](jetson-5tier-experiment.md)と同じ7台・同じトポロジーのまま、**1日以上動き続けるか**を検証する。
そのために次を用意した。

| 要素 | 実体 |
|---|---|
| 1コマンド起動・停止・状態確認 | `scripts/hfl.sh`（各Jetson上でも、.201やPCからの一括操作でも使う） |
| 実質無限の学習と時間上限 | `config/jetson-5tier-longrun/`（`num_rounds: 1000000`、`max_duration_hours: 240`） |
| 証拠の記録 | 各ノードの `logs/<run-id>/events-<NUM>.jsonl`（1行1イベント、タイムスタンプ付き、電源断に備え行ごとに fsync） |
| リアルタイム監視 | `scripts/hfl.sh monitor start` → ブラウザで `http://192.168.10.201:8000/` |
| 回収と可視化 | `scripts/hfl.sh collect` → `scripts/plot_run.py`（CSV + PNG） |

脱落時の挙動は **fail-fast のまま**（1台でも落ちると全体が止まる）。[§9](#9-脱落時の挙動と注意) を参照。

## 2. 事前準備

1. **配備**: いつもどおり `.\scripts\deploy_jetsons.ps1 -IdentityFile "C:\Users\iamzo\.ssh\id_ed25519"`（PCから。HEAD のコミットが7台に入る）。
2. **CIFAR-10 のキャッシュ**: 3層実験手順書の §3 のとおり、各Jetsonに `~/.cache/huggingface/datasets/uoft-cs___cifar10` があること。`hfl.sh` は Hugging Face をオフラインモード（`HF_DATASETS_OFFLINE=1`）で起動するので、キャッシュが無いノードは即座に失敗する（`start-all` が事前に確認する）。
3. **UFW**: 3層・5段実験と同じ（.201: 8080、.202: 9001、.203: 9002、.204: 9003、.206: 9004）。
4. **.201 から一括操作する場合の SSH 鍵**（一度だけ）:
   ```bash
   # .201 で
   ssh-keygen -t ed25519            # 既にあれば不要
   for n in 2 3 4 5 6 7; do ssh-copy-id tachilab-orin0$n@192.168.10.20$n; done   # 6回パスワード入力
   ```
   PC の Git Bash から操作する場合は既存の鍵をそのまま使う（`--identity "C:/Users/iamzo/.ssh/id_ed25519"` または環境変数 `HFL_SSH_KEY`）。
5. **時計**: 実験 LAN にはインターネットが無く NTP が効かないので、各Jetsonの時計はずれうる。`scripts/hfl.sh clock-check` でずれを確認し、大きければ `sudo date -s "..."` で合わせる。`start-all` と `collect` もずれを記録する（可視化時に補正できる）。

## 3. 台帳とプロファイル

`config/jetson-5tier-longrun/` に設定一式がある。

| ファイル | 内容 |
|---|---|
| `nodes.yaml` | 7台の台帳（番号・役割・ポート・親）。Edge の設定ファイルは `topology.yaml`、Leaf の partition は `assignments` から導く |
| `global.yaml` | `num_rounds: 1000000`、`max_duration_hours: 240`、`round_timeout: 1500` |
| `edge_01/02.yaml`, `leaf_01/03.yaml` | 5段実験と同じ構成。タイムアウトは下表 |
| `topology.yaml` | 5段実験と同じ親子関係・Partition |

タイムアウト連鎖（秒）。実測 1ラウンド ≈ 18 秒（初回 ≈ 140 秒）に対し十分な余裕を取りつつ、固まった端末を何十分も待たない値。

| ノード | sub_round_timeout | parent_result_timeout | 直上の round_timeout | parent_model_timeout |
|---|---:|---:|---:|---:|
| Global | — | — | (round_timeout 1500) | — |
| edge_01 / edge_02 | 1200 | 1260 | 1500 (Global) | 2400 |
| leaf_01 | 900 | 960 | 1200 (edge_01) | 2400 |
| leaf_03 | 600 | 660 | 900 (leaf_01) | 2400 |

ルール: `sub_round_timeout < parent_result_timeout < 直上の round_timeout`、`parent_model_timeout > Global round_timeout + 600`。`parent_model_timeout` の条件は起動時にも検証される（満たさないと `ValueError`）。

短時間の実験には `--profile config/jetson-5tier`（5ラウンド）も同じコマンドで使える。

`scripts/hfl.sh check` で台帳と設定ファイルの整合性を検証できる。

## 4. 起動

### A. 各Jetsonで1コマンド（PC 不要）

7台にそれぞれ SSH で入り、**Global → Edge 01/02 → L01 → L02 → L03 → L04 の順**に実行する。
番号は IP から自動判定されるので省略できる。`--run-id` は7台で同じ文字列にする。

```bash
cd ~/hierarchical-fl-persistent
scripts/hfl.sh start --run-id longrun-20261011-01 --max-hours 240   # .201 (Global)
scripts/hfl.sh status                                                 # port 8080: listening を確認
```
```bash
scripts/hfl.sh start --run-id longrun-20261011-01   # .202, .203 (Edge)。それぞれ status で 9001/9002 の待受を確認
scripts/hfl.sh start --run-id longrun-20261011-01   # .204 (L01)。9003 を確認
scripts/hfl.sh start --run-id longrun-20261011-01   # .205 (L02)
scripts/hfl.sh start --run-id longrun-20261011-01   # .206 (L03)。9004 を確認
scripts/hfl.sh start --run-id longrun-20261011-01   # .207 (L04)
```

`--max-hours` は Global だけに意味がある（省略時は `global.yaml` の `max_duration_hours` = 240 時間）。
`start` は既に HFL プロセスが動いていれば拒否する。バックグラウンドで動くので SSH を切ってよい。

### B. .201（または PC の Git Bash）から7台を一括起動

```bash
cd ~/hierarchical-fl-persistent
scripts/hfl.sh start-all --run-id longrun-20261011-01 --max-hours 240
```

親の待受を `ss -ltn` で確認してから子を起動する（PowerShell 版と同じ順序）。起動前に全台の既存プロセス・データセットキャッシュ・時計ずれを確認する。
疎通確認だけなら `--dry-run` を付ける（dummy データで2ラウンド回って自動終了）。
PC の Git Bash から実行する場合は `--identity "C:/Users/iamzo/.ssh/id_ed25519"` を付ける（または `export HFL_SSH_KEY=...`）。

## 5. 監視

```bash
scripts/hfl.sh monitor start --run-id longrun-20261011-01      # .201 で
scripts/hfl.sh status-all    --run-id longrun-20261011-01      # 端末に1表
scripts/hfl.sh wait          --run-id longrun-20261011-01      # Global 終了までブロック
```

監視は 15 秒ごとに7台へ SSH し、`logs/<run-id>/monitor/status.json` を更新する。同じディレクトリを HTTP で配信するので、同じ LAN の PC のブラウザで `http://192.168.10.201:8000/` を開く（監視を PC で動かした場合は `http://localhost:8000/`）。

ページに出るもの: 生存台数、Global ラウンド、accuracy / loss、各ノードの状態（稼働中 / 更新停止 / 停止 / 到達不能）、PID、稼働時間、メモリ、待受ポート、最終イベントからの経過、時計ずれ、Traceback 数、Global accuracy / loss の時系列、各ノードのラウンド進行の時系列、ログ末尾。

「サーバが生存を把握できるか」について: Global が直接知るのは直下の Edge 2台だけ。各階層の `aggregate` イベント（誰の結果を集約したか）と各台のプロセス生存を監視が合成して全体像にしている。

## 6. 停止

```bash
scripts/hfl.sh stop-all --run-id longrun-20261011-01     # .201 から（推奨）
scripts/hfl.sh stop     --run-id longrun-20261011-01     # 各台で個別に
```

- Global の `stop` は `logs/<run-id>/STOP` を作る。Global は**現在のラウンドを終えてから**停止し、切断通知が Edge → 中継 → Leaf へ連鎖して全プロセスが正常終了する（Traceback なし、終了コード 0）。時間上限 (`max_duration`) も同じ経路。
- Global 以外の `stop` は即時終了（プロセスグループごと TERM → KILL）。fail-fast なので全体が止まる。
- 各ノードの終了コードは `logs/<run-id>/hfl-<NUM>.exit` に残る。このファイルが無い = 強制終了か電源断。

## 7. 回収と可視化

```bash
scripts/hfl.sh collect --run-id longrun-20261011-01          # collected/<run-id>/<NUM>/ に全台の logs を scp
```
PC で:
```bash
uv sync --extra analysis                                       # matplotlib（一度だけ）
uv run python scripts/plot_run.py collected/longrun-20261011-01
```

出力（`collected/<run-id>/plots/`）:

| ファイル | 内容 |
|---|---|
| `events.csv` | 全ノードの全イベントを1本にしたもの（Excel 等でも読める） |
| `summary.json` | ラウンド数、稼働時間、精度、各ノードの fit 回数・所要時間・停止理由、長い空白区間 |
| `timeline_rounds.png` | 各ノードの Global ラウンド番号 vs 時刻。止まった端末は線が途切れる |
| `global_metrics.png` / `global_metrics_vs_round.png` | Global accuracy / loss |
| `participation.png` | ノード × ラウンドの参加マトリクス |
| `fit_duration.png` | 各クライアントの fit 所要時間。遅い端末の特定に使う |
| `round_duration.png` | Global ラウンド所要時間 |

イベントの形式（JSONL、1行1イベント）。主なフィールド: `ts`（ISO 8601、タイムゾーン付き）、`t`（epoch 秒）、`run_id`、`node`、`node_num`、`host`、`proc`（global / edge / sub_server / internal_client / leaf）、`event`、`global_round`、`parent_round`、`child_round`、`duration_s`、`loss`、`accuracy`、`participants`、`reason`。
`event` の種類: `start` / `stop` / `error`（全プロセス）、`round_start` / `aggregate` / `eval`（Global と sub-server）、`parent_model_received` / `child_result_sent`（sub-server）、`parent_fit_start` / `parent_fit_end`（Edge）、`fit_start` / `fit_end`（Leaf と Internal）。

## 8. 成功判定

1. Global の events に `stop` があり `reason` が `max_duration`（時間上限）または `stop_file`（手動停止）。
2. 全ノードの `hfl-<NUM>.exit` が `0`。events の最後の `stop` の `reason` が、Edge/中継は `finished`（edge）・`stop_requested`（sub_server）・`finished`（internal_client）、Leaf は `finished`。
3. `error` イベントと、ログの `Traceback` が無い。
4. `timeline_rounds.png` で7台の線が同じラウンドまで途切れずに伸びている。
5. 終了後、全台の `pgrep -af '[s]rc.core'` に残りが無い。

## 9. 脱落時の挙動と注意

- **fail-fast**: どの階層でも子の結果が欠けると親まで連鎖的に止まる。電源が落ちた端末は復帰しても再合流できない（全体を `stop-all` → `start-all` し直す）。「何時間・何ラウンド持ったか」と「どの端末が最初に落ちたか」（`timeline_rounds.png`、各台の `stop`/`error` イベントの時刻）が証拠になる。
- **Global (.201) は単一障害点**: .201 が落ちると全体が終わる。チェックポイントはない。
- **ノード個別の停止は全体停止**: 実験中に1台だけ `stop` すると fail-fast で全体が止まる。
- **ログ容量**: 1ラウンド ≈ 18 秒で10日回すと約 48,000 ラウンド。ログは1台あたり 100 MB 前後、events は 50 MB 前後になる。
- **時計**: 複数台の時刻をまたぐ図は各台の時計に依存する。`clock-check` と `collect` の `clock_offsets.json` で補正する。

遅い端末への方針: fail-fast では「全端末が毎ラウンド参加し、ラウンド時間は最遅端末に律速、タイムアウトは誤検知しない程度に大きく」。`fit_duration.png` で遅い端末を特定してから、必要なら「締切までに届いた分で集約する」モード（未実装、`global.yaml` の `fault_tolerance` として追加予定）を検討する。

## 10. 1PC での確認

```bash
uv run python run.py --dry-run --topology-config config/local-5tier/topology.yaml          # 2ラウンド疎通
uv run python run.py --dry-run --global-config config/local-5tier/global_longrun.yaml \
    --topology-config config/local-5tier/topology.yaml --run-id local-maxhours            # 108秒で時間上限停止
uv run python scripts/plot_run.py logs/local-maxhours
```

`tests/test_integration.py::test_max_hours_stops_all_processes_cleanly` が2つ目を自動化している（全プロセス exit 0、`max_duration`、Traceback なし、events の `stop` 理由）。

## 11. トラブルシュート

| 症状 | 対処 |
|---|---|
| `start` が "already running" で拒否 | `pgrep -af '[s]rc.core'` で確認し、前回の run-id で `scripts/hfl.sh stop` |
| 起動直後に exit 1、ログに `ValueError ... parent_model_timeout` | 設定のタイムアウト連鎖が崩れている。§3 のルールに合わせる |
| 起動直後に `Dataset ... not found`（HF offline） | CIFAR-10 のキャッシュが無い。§2 の手順でダウンロード |
| `stop` しても Global が止まらない | ラウンド境界でしか止まらない。固まっている場合は round_timeout (1500 s) 後に止まる。急ぐなら再度 `stop`（STOP_TIMEOUT 経過後に強制終了） |
| 監視ページが開けない | `scripts/hfl.sh monitor status`。.201 の UFW で 8000 を開ける: `sudo ufw allow from 192.168.10.0/24 to any port 8000 proto tcp` |
| 新しい run-id なのに即停止する | 前の `STOP` ファイルが残っている。`start` は同じ run-id の STOP を消すが、別 run-id の場合は無関係 |
