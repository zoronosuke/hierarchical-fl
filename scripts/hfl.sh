#!/usr/bin/env bash
# =============================================================================
# Jetson 7台 HFL 運用コマンド (1コマンド起動・停止・状態確認・ログ回収・監視)
#
#   各 Jetson 上で自分の役割だけ起動:   scripts/hfl.sh start            (NUM は IP から自動判定)
#   .201 や PC から 7 台まとめて起動:   scripts/hfl.sh start-all --max-hours 240
#
# 台帳は config/<profile>/nodes.yaml (既定: config/jetson-5tier-longrun)。
# 詳細は scripts/hfl.sh help と docs/jetson-longrun-experiment.md を参照。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${HFL_PROFILE:-config/jetson-5tier-longrun}"
RUN_ID="${HFL_RUN_ID:-}"
NUM=""
DRY_RUN=0
MAX_HOURS=""
ROUND_INTERVAL=""
RESUME=0
FORCE=0
IDENTITY="${HFL_SSH_KEY:-}"
OUT_DIR=""
STARTUP_TIMEOUT="${HFL_STARTUP_TIMEOUT:-180}"
STOP_TIMEOUT="${HFL_STOP_TIMEOUT:-300}"
RESUME_WAIT="${HFL_RESUME_WAIT:-1800}"   # resume で全ノードの到達を待つ上限秒数
MONITOR_INTERVAL="${HFL_MONITOR_INTERVAL:-15}"
LD_PATH_JETSON="/usr/lib/aarch64-linux-gnu/libcudss/12"

usage() {
    cat <<'EOF'
Usage: scripts/hfl.sh <command> [NUM] [options]

このノードで実行するコマンド (NUM 省略時は自ホストの IP から判定):
  start   [NUM]   役割 (global / edge / leaf) をバックグラウンドで起動
  stop    [NUM]   停止 (global はラウンド境界で正常停止 → 配下へ連鎖。他は即時。--force で即時)
  status  [NUM]   プロセス・待受ポート・最終イベント・記録プロセスを表示
  sampler-stop [NUM]  Jetson 側の記録プロセス (電力・温度・通信量) を止める (stop-all が呼ぶ)

全ノードに対して SSH で実行するコマンド (.201 や PC から):
  start-all       親→子の順に全ノードを起動 (親の待受を確認してから子を起動)
  stop-all        Global を正常停止し、連鎖終了を待ってから残りを停止
  resume          止まった実験を同じ run-id でチェックポイントの続きから再開
                  (残ったプロセスを片付け → 全ノードの到達を待つ → 親→子の順に起動)
  status-all      全ノードの状態を1表で表示
  wait            Global が終了するまで待つ
  collect         全ノードの logs/<run-id>/ を collected/<run-id>/<NUM>/ に回収
  clock-check     各ノードの時計ずれを表示
  monitor start|stop|status   監視ページ (http://<このホスト>:8000/) を起動/停止
  check           台帳と設定ファイルの整合性を検証

Options:
  --profile DIR     台帳・設定のディレクトリ (既定: config/jetson-5tier-longrun)
  --run-id ID       実験 ID (既定: start/start-all は日時、他は logs/current_run)
  --dry-run         疎通確認モード (dummy データ・2ラウンド)
  --max-hours H     Global の時間上限 (YAML の max_duration_hours を上書き。再開後も最初の開始から数える)
  --resume          start / start-all で Global をチェックポイントの続きから起動
  --round-interval S  Global の学習間隔を一時的に短くする (YAML の round_interval_s 以下)
  --force           stop で正常停止を待たずに即時終了
  --identity FILE   SSH 秘密鍵 (環境変数 HFL_SSH_KEY でも可)
  --out DIR         collect の出力先
  --interval SEC    monitor のポーリング間隔 (既定 15)
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[hfl] $*"; }

# ---------------------------------------------------------------------------
# 引数解析
# ---------------------------------------------------------------------------
[[ $# -ge 1 ]] || { usage; exit 1; }
COMMAND="$1"; shift
SUBCOMMAND=""
if [[ "$COMMAND" == "monitor" ]]; then
    [[ $# -ge 1 ]] || die "monitor には start|stop|status を指定してください"
    SUBCOMMAND="$1"; shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --round-interval) ROUND_INTERVAL="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --identity) IDENTITY="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --interval) MONITOR_INTERVAL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        [0-9][0-9][0-9]) NUM="$1"; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ -d "$PROFILE" ]] || die "profile directory not found: $PROFILE"
if [[ -n "$RUN_ID" && ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "run-id may contain only letters, digits, dot, underscore, and hyphen"
fi

# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------
find_python() {
    if [[ -n "${HFL_PYTHON:-}" ]]; then echo "$HFL_PYTHON"; return; fi
    local candidate
    for candidate in .venv/bin/python .venv/Scripts/python.exe; do
        if [[ -x "$candidate" ]]; then echo "$candidate"; return; fi
    done
    echo "python"
}
PY="$(find_python)"
nodes() { "$PY" scripts/hfl_nodes.py --profile "$PROFILE" "$@"; }

eval "$(nodes settings)"   # HFL_REMOTE_DIR, HFL_MONITOR_PORT, HFL_NO_PROXY_LIST, HFL_NODE_NUMS ...

declare -A NODE_IP NODE_USER NODE_ROLE NODE_ID NODE_PORT
while IFS=$'\t' read -r num ip user role id port _rest; do
    NODE_IP[$num]="$ip"; NODE_USER[$num]="$user"; NODE_ROLE[$num]="$role"
    NODE_ID[$num]="$id"; NODE_PORT[$num]="$port"
done < <(nodes list)

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
[[ -n "$IDENTITY" ]] && SSH_OPTS+=(-i "$IDENTITY")

SELF_NUM="$(nodes self 2>/dev/null || true)"
is_self() { [[ -n "$SELF_NUM" && "$SELF_NUM" == "$1" ]]; }
target() { echo "${NODE_USER[$1]}@${NODE_IP[$1]}"; }

# remote NUM <command...>: 自ノードならローカルで、他ノードなら SSH で実行する
remote() {
    local num="$1"; shift
    if is_self "$num"; then
        bash -c "cd '$REPO_ROOT' && $*"
    else
        ssh "${SSH_OPTS[@]}" "$(target "$num")" "cd $HFL_REMOTE_DIR && $*"
    fi
}

# 他ノード上の hfl.sh を呼ぶ (実行ビットが無くても動くよう bash 経由)
remote_hfl() {
    local num="$1"; shift
    remote "$num" "bash scripts/hfl.sh $*"
}

require_num() {
    if [[ -z "$NUM" ]]; then
        NUM="$SELF_NUM"
        [[ -n "$NUM" ]] || die "NUM を指定してください (このホストは $PROFILE/nodes.yaml にありません)"
    fi
    [[ -n "${NODE_ROLE[$NUM]:-}" ]] || die "node $NUM is not in $PROFILE/nodes.yaml"
}

current_run_id() {
    if [[ -z "$RUN_ID" ]]; then
        [[ -f logs/current_run ]] || die "--run-id を指定してください (logs/current_run がありません)"
        RUN_ID="$(cat logs/current_run)"
    fi
}

pid_alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }

# 監視ページの URL 表示用。Linux は hostname -I、Git Bash (Windows) は localhost
local_ip() {
    local ip
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "${ip:-localhost}"
}

read_pid() {  # read_pid NUM -> pid or ""
    local pidfile="logs/$RUN_ID/hfl-$1.pid"
    [[ -f "$pidfile" ]] && cat "$pidfile" || true
}

read_exit() {  # read_exit NUM -> exit code or "" (強制終了時は記録されない)
    local exitfile="logs/$RUN_ID/hfl-$1.exit"
    [[ -f "$exitfile" ]] && cat "$exitfile" || true
}

# プロセスグループ (python + Edge の子プロセス) の RSS 合計 (MB)
group_rss_mb() {
    ps -eo pgid=,rss= 2>/dev/null | awk -v g="$1" '$1==g{s+=$2} END{printf "%d", s/1024}'
}

# ---------------------------------------------------------------------------
# Jetson 側の記録プロセス (scripts/hfl_sampler.py)
# 台帳に sampler_interval_s がある profile だけで使う。学習とは別に動き、学習が落ちても
# 記録を続ける。stop / stop --force (resume の片付け) は触らず、stop-all が最後に止める。
# ---------------------------------------------------------------------------
sampler_pid_ok() {  # sampler_pid_ok PID RUN_ID: そのPIDがこの run の記録プロセスか
    local pid="$1" run="$2"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 1
    # 電源断後の再起動で同じ PID が別のプロセスに使われていないか、コマンドラインで確かめる
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "hfl_sampler.py .*--run-id $run "
}

start_sampler() {
    [[ -n "$HFL_SAMPLER_INTERVAL_S" ]] || return 0
    local pidfile="logs/$RUN_ID/sampler-$NUM.pid" pid other
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if sampler_pid_ok "$pid" "$RUN_ID"; then
        log "sampler already running (pid=$pid)"
        return 0
    fi
    # 1台に記録プロセスは1つ: 別の run のものが残っていれば止める
    for other in $(pgrep -f '[h]fl_sampler.py' || true); do
        log "stopping a sampler of another run (pid=$other)"
        kill "$other" 2>/dev/null || true
    done
    local launcher=(bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pidfile"
                    "$PY" scripts/hfl_sampler.py --profile "$PROFILE" --run-id "$RUN_ID"
                    --node-num "$NUM" --interval "$HFL_SAMPLER_INTERVAL_S")
    if command -v setsid >/dev/null 2>&1; then
        setsid "${launcher[@]}" >> "logs/$RUN_ID/sampler-$NUM.log" 2>&1 < /dev/null &
    else
        "${launcher[@]}" >> "logs/$RUN_ID/sampler-$NUM.log" 2>&1 < /dev/null &
    fi
    sleep 1
    log "sampler started (pid=$(cat "$pidfile" 2>/dev/null), every ${HFL_SAMPLER_INTERVAL_S}s -> logs/$RUN_ID/metrics-$NUM.jsonl)"
}

cmd_sampler_stop() {
    require_num
    current_run_id
    local pid; pid="$(cat "logs/$RUN_ID/sampler-$NUM.pid" 2>/dev/null || true)"
    if sampler_pid_ok "$pid" "$RUN_ID"; then
        kill "$pid" && log "node $NUM: sampler stopped (pid=$pid)"
    else
        log "node $NUM: sampler is not running"
    fi
}

# ---------------------------------------------------------------------------
# このノードで実行するコマンド
# ---------------------------------------------------------------------------
cmd_start() {
    require_num
    if pgrep -f '[s]rc\.core' >/dev/null 2>&1; then
        pgrep -af '[s]rc\.core' >&2
        die "HFL process is already running on this host (stop it first)"
    fi
    RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
    eval "$(nodes node "$NUM")"   # NODE_NUM NODE_ROLE NODE_ID NODE_PORT ...
    mkdir -p "logs/$RUN_ID"
    echo "$RUN_ID" > logs/current_run
    rm -f "logs/$RUN_ID/STOP" "logs/$RUN_ID/hfl-$NUM.exit"
    git rev-parse --short HEAD > "logs/$RUN_ID/revision" 2>/dev/null || true
    start_sampler

    local cmd_flags=()
    [[ "$DRY_RUN" == 1 ]] && cmd_flags+=(--dry-run)
    [[ -n "$MAX_HOURS" ]] && cmd_flags+=(--max-hours "$MAX_HOURS")
    [[ "$RESUME" == 1 ]] && cmd_flags+=(--resume)
    [[ -n "$ROUND_INTERVAL" ]] && cmd_flags+=(--round-interval "$ROUND_INTERVAL")
    local cmd=()
    mapfile -t cmd < <(nodes command "$NUM" "${cmd_flags[@]}")

    local envs=(
        PYTHONUNBUFFERED=1
        HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
        NO_PROXY="$HFL_NO_PROXY_LIST" no_proxy="$HFL_NO_PROXY_LIST" no_grpc_proxy="$HFL_NO_PROXY_LIST"
        HFL_EVENT_LOG="logs/$RUN_ID/events-$NUM.jsonl"
        HFL_STOP_FILE="logs/$RUN_ID/STOP"
        HFL_CHECKPOINT="logs/$RUN_ID/checkpoint.npz"
        HFL_RUN_ID="$RUN_ID" HFL_NODE_ID="$NODE_ID" HFL_NODE_NUM="$NUM"
    )
    if [[ -d "$LD_PATH_JETSON" ]]; then
        envs+=(LD_LIBRARY_PATH="$LD_PATH_JETSON:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    fi
    local logfile="logs/$RUN_ID/hfl-$NUM.log" pidfile="logs/$RUN_ID/hfl-$NUM.pid"
    local exitfile="logs/$RUN_ID/hfl-$NUM.exit"
    log "start $NODE_ROLE $NODE_ID on $NUM (run_id=$RUN_ID, log=$logfile)"
    log "  $PY ${cmd[*]}"
    # ラッパー bash が自分の PID (= setsid 後のプロセスグループ ID) を書き、python の終了コードを
    # .exit に残す。強制終了されたときは .exit が無い = 正常終了ではない証拠になる。
    local launcher=(bash -c 'echo $$ > "$1"; exitfile="$2"; shift 2; "$@"; code=$?; echo $code > "$exitfile"; exit $code'
                    _ "$pidfile" "$exitfile" env "${envs[@]}" "$PY" "${cmd[@]}")
    # ログは追記する (同じ run-id で再起動・再開しても前のログを消さない)
    echo "===== [hfl] start $(date '+%Y-%m-%d %H:%M:%S') node=$NUM resume=$RESUME =====" >> "$logfile"
    if command -v setsid >/dev/null 2>&1; then
        setsid "${launcher[@]}" >> "$logfile" 2>&1 < /dev/null &
    else
        "${launcher[@]}" >> "$logfile" 2>&1 < /dev/null &
    fi
    sleep 3
    local pid; pid="$(read_pid "$NUM")"
    if ! pid_alive "$pid"; then
        tail -n 20 "$logfile" >&2 || true
        die "process exited right after start (exit=$(read_exit "$NUM"), see $logfile)"
    fi
    log "started pid=$pid"
}

stop_pid_group() {  # stop_pid_group PID GRACE_SECONDS
    local pid="$1" grace="$2" waited=0
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    while pid_alive "$pid" && (( waited < grace )); do sleep 1; waited=$((waited + 1)); done
    if pid_alive "$pid"; then
        log "pid $pid still alive after ${grace}s: sending KILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
}

cmd_stop() {
    require_num
    current_run_id
    local pid; pid="$(read_pid "$NUM")"
    if ! pid_alive "$pid"; then
        log "node $NUM (${NODE_ID[$NUM]}) is not running (run_id=$RUN_ID)"
        return 0
    fi
    if [[ "${NODE_ROLE[$NUM]}" == "global" && "$FORCE" != 1 ]]; then
        # 停止ファイル → Global が次のラウンド境界で止まり、配下へ切断通知が連鎖する
        log "global pid=$pid: requesting stop at the next round boundary (logs/$RUN_ID/STOP)"
        touch "logs/$RUN_ID/STOP"
        local waited=0
        while pid_alive "$pid" && (( waited < STOP_TIMEOUT )); do sleep 2; waited=$((waited + 2)); done
        if pid_alive "$pid"; then
            log "global did not stop within ${STOP_TIMEOUT}s: terminating"
            stop_pid_group "$pid" 20
        fi
    else
        log "node $NUM (${NODE_ID[$NUM]}) pid=$pid: terminating process group"
        stop_pid_group "$pid" 30
    fi
    log "stopped (exit=$(read_exit "$NUM"))"
}

cmd_status() {
    require_num
    current_run_id
    local pid; pid="$(read_pid "$NUM")"
    echo "node=$NUM id=${NODE_ID[$NUM]} role=${NODE_ROLE[$NUM]} run_id=$RUN_ID host=$(hostname)"
    if pid_alive "$pid"; then
        local up; up="$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')"
        echo "process: running pid=$pid rss=$(group_rss_mb "$pid")MB up=${up:-?}s"
    else
        echo "process: NOT running (pid=${pid:-none}, exit=$(read_exit "$NUM"))"
    fi
    local port="${NODE_PORT[$NUM]}"
    if [[ "$port" != "-" ]]; then
        if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "port $port: listening"; else echo "port $port: closed"; fi
    fi
    local events="logs/$RUN_ID/events-$NUM.jsonl"
    if [[ -f "$events" ]]; then
        echo "last event: $(tail -n 1 "$events" | cut -c1-300)"
    fi
    local logfile="logs/$RUN_ID/hfl-$NUM.log"
    [[ -f "$logfile" ]] && echo "last log: $(tail -n 1 "$logfile" | cut -c1-200)"
    if [[ -n "$HFL_SAMPLER_INTERVAL_S" ]]; then
        local spid; spid="$(cat "logs/$RUN_ID/sampler-$NUM.pid" 2>/dev/null || true)"
        if sampler_pid_ok "$spid" "$RUN_ID"; then
            echo "sampler: running pid=$spid (logs/$RUN_ID/metrics-$NUM.jsonl)"
        else
            echo "sampler: NOT running"
        fi
    fi
    return 0
}

# ---------------------------------------------------------------------------
# 全ノードに対するコマンド
# ---------------------------------------------------------------------------
wait_listener() {  # wait_listener NUM PORT
    local num="$1" port="$2" waited=0
    while (( waited < STARTUP_TIMEOUT )); do
        if remote "$num" "ss -ltn | grep -q ':$port '" 2>/dev/null; then
            log "verified listener ${NODE_IP[$num]}:$port"
            return 0
        fi
        if ! remote "$num" "kill -0 \$(cat logs/$RUN_ID/hfl-$num.pid 2>/dev/null) 2>/dev/null"; then
            remote "$num" "tail -n 20 logs/$RUN_ID/hfl-$num.log" >&2 || true
            die "node $num exited before listening on $port"
        fi
        sleep 2; waited=$((waited + 2))
    done
    die "timed out waiting for ${NODE_IP[$num]}:$port (check $HFL_REMOTE_DIR/logs/$RUN_ID/)"
}

cmd_check() {
    nodes check
}

cmd_clock_check() {  # 出力: NUM<TAB>offset_s<TAB>rtt_s  (offset = ノード時計 - このホストの時計)
    local num t0 t1 tr offset rtt
    for num in $HFL_NODE_NUMS; do
        t0="$(date +%s.%N)"
        if tr="$(remote "$num" 'date +%s.%N' 2>/dev/null)"; then
            t1="$(date +%s.%N)"
            offset="$(awk -v a="$t0" -v b="$t1" -v r="$tr" 'BEGIN{printf "%.3f", r-(a+b)/2}')"
            rtt="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')"
            printf '%s\t%s\t%s\n' "$num" "$offset" "$rtt"
        else
            printf '%s\tunreachable\t-\n' "$num"
        fi
    done
}

cmd_start_all() {
    nodes check >/dev/null
    RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
    log "pre-check: existing HFL processes"
    local num busy=0 out
    for num in $HFL_NODE_NUMS; do
        out="$(remote "$num" "pgrep -af '[s]rc\\.core' || true" 2>&1)" || die "cannot reach node $num: $out"
        if [[ -n "$out" ]]; then echo "  node $num: $out" >&2; busy=1; fi
    done
    (( busy == 0 )) || die "HFL processes are still running; run stop-all first"
    if [[ "$DRY_RUN" != 1 ]]; then
        log "pre-check: CIFAR-10 cache (HF offline mode)"
        for num in $HFL_NODE_NUMS; do
            remote "$num" "test -d ~/.cache/huggingface/datasets/uoft-cs___cifar10" \
                || die "node $num: dataset cache missing (see docs: CIFAR-10 の準備)"
        done
    fi

    log "clock offsets (node - this host, seconds)"
    cmd_clock_check | while IFS=$'\t' read -r num offset rtt; do
        local warn=""
        if [[ "$offset" != "unreachable" ]] && awk -v o="$offset" 'BEGIN{exit !(o > 2 || o < -2)}'; then warn="  <-- WARNING: >2s"; fi
        echo "  node $num: offset=${offset}s rtt=${rtt}s$warn"
    done

    local flags=(--profile "$PROFILE" --run-id "$RUN_ID")
    [[ "$DRY_RUN" == 1 ]] && flags+=(--dry-run)
    [[ -n "$MAX_HOURS" ]] && flags+=(--max-hours "$MAX_HOURS")
    [[ "$RESUME" == 1 ]] && flags+=(--resume)
    [[ -n "$ROUND_INTERVAL" ]] && flags+=(--round-interval "$ROUND_INTERVAL")
    mkdir -p logs; echo "$RUN_ID" > logs/current_run
    for num in $HFL_NODE_NUMS; do
        log "== node $num (${NODE_ROLE[$num]} ${NODE_ID[$num]})"
        remote_hfl "$num" "start $num ${flags[*]}"
        if [[ "${NODE_PORT[$num]}" != "-" ]]; then wait_listener "$num" "${NODE_PORT[$num]}"; fi
    done
    log "all nodes launched. run_id=$RUN_ID"
    log "logs/events: $HFL_REMOTE_DIR/logs/$RUN_ID/ on each node"
    log "next: scripts/hfl.sh monitor start --run-id $RUN_ID   (http://$(local_ip):$HFL_MONITOR_PORT/)"
}

any_alive() {  # 1つでも動いていれば 0
    local num pid_cmd
    for num in $HFL_NODE_NUMS; do
        pid_cmd="kill -0 \$(cat logs/$RUN_ID/hfl-$num.pid 2>/dev/null) 2>/dev/null"
        if remote "$num" "$pid_cmd" 2>/dev/null; then return 0; fi
    done
    return 1
}

global_num() {
    local num
    for num in $HFL_NODE_NUMS; do
        if [[ "${NODE_ROLE[$num]}" == "global" ]]; then echo "$num"; return; fi
    done
}

# 止まった実験を、同じ run-id でチェックポイントの続きから再開する
cmd_resume() {
    current_run_id
    local gnum num pending waited=0
    gnum="$(global_num)"
    log "resume run_id=$RUN_ID: waiting for all nodes to be reachable (up to ${RESUME_WAIT}s)"
    while :; do
        pending=""
        for num in $HFL_NODE_NUMS; do
            remote "$num" "true" >/dev/null 2>&1 || pending+=" $num"
        done
        [[ -z "$pending" ]] && break
        (( waited >= RESUME_WAIT )) && die "nodes still unreachable:$pending"
        log "  unreachable:$pending (waited ${waited}s)"
        sleep 15; waited=$((waited + 15))
    done
    remote "$gnum" "test -f logs/$RUN_ID/checkpoint.npz" \
        || die "no checkpoint on node $gnum: logs/$RUN_ID/checkpoint.npz"

    log "cleaning up leftover processes of run_id=$RUN_ID"
    for num in $HFL_NODE_NUMS; do
        remote_hfl "$num" "stop $num --force --profile $PROFILE --run-id $RUN_ID" | sed 's/^/  /' \
            || log "node $num: cleanup failed"
    done
    log "starting all nodes from the checkpoint"
    RESUME=1
    cmd_start_all
}

cmd_stop_all() {
    current_run_id
    local global_num
    for global_num in $HFL_NODE_NUMS; do [[ "${NODE_ROLE[$global_num]}" == "global" ]] && break; done
    log "requesting graceful stop on global ($global_num)"
    remote_hfl "$global_num" "stop $global_num --profile $PROFILE --run-id $RUN_ID" || true
    log "waiting up to ${STOP_TIMEOUT}s for the shutdown cascade"
    local waited=0
    while any_alive && (( waited < STOP_TIMEOUT )); do sleep 5; waited=$((waited + 5)); done
    local num
    for num in $HFL_NODE_NUMS; do
        remote_hfl "$num" "stop $num --profile $PROFILE --run-id $RUN_ID" || log "node $num: stop failed"
    done
    log "exit codes (empty = killed, not a clean exit):"
    for num in $HFL_NODE_NUMS; do
        echo "  node $num: exit=$(remote "$num" "cat logs/$RUN_ID/hfl-$num.exit 2>/dev/null" || true)"
    done
    if [[ -n "$HFL_SAMPLER_INTERVAL_S" ]]; then
        log "stopping the Jetson-side samplers"
        for num in $HFL_NODE_NUMS; do
            remote_hfl "$num" "sampler-stop $num --profile $PROFILE --run-id $RUN_ID" \
                || log "node $num: sampler-stop failed"
        done
    fi
    log "stop-all done (run_id=$RUN_ID)"
}

cmd_status_all() {
    current_run_id
    local args=(--profile "$PROFILE" --run-id "$RUN_ID" --once --table)
    [[ -n "$IDENTITY" ]] && args+=(--identity "$IDENTITY")
    "$PY" scripts/hfl_monitor.py "${args[@]}"
}

cmd_wait() {
    current_run_id
    local global_num
    for global_num in $HFL_NODE_NUMS; do [[ "${NODE_ROLE[$global_num]}" == "global" ]] && break; done
    log "waiting for global ($global_num) to finish (run_id=$RUN_ID)"
    while remote "$global_num" "kill -0 \$(cat logs/$RUN_ID/hfl-$global_num.pid 2>/dev/null) 2>/dev/null"; do
        remote "$global_num" "grep '\"event\": \"eval\"' logs/$RUN_ID/events-$global_num.jsonl 2>/dev/null | tail -n 1 | cut -c1-200" || true
        sleep 30
    done
    remote "$global_num" "tail -n 5 logs/$RUN_ID/hfl-$global_num.log" || true
    log "global finished"
}

cmd_collect() {
    current_run_id
    local out="${OUT_DIR:-collected/$RUN_ID}" num
    mkdir -p "$out"
    for num in $HFL_NODE_NUMS; do
        mkdir -p "$out/$num"
        if is_self "$num"; then
            cp -r "logs/$RUN_ID/." "$out/$num/" && log "node $num: copied locally"
        elif scp -r -q "${SSH_OPTS[@]}" "$(target "$num"):$HFL_REMOTE_DIR/logs/$RUN_ID/." "$out/$num/"; then
            log "node $num: collected"
        else
            log "node $num: FAILED (unreachable or no logs/$RUN_ID)"
        fi
    done
    nodes json > "$out/nodes.json"
    {
        echo "{"
        local first=1
        while IFS=$'\t' read -r num offset rtt; do
            (( first )) || echo ","
            first=0
            if [[ "$offset" == "unreachable" ]]; then printf '  "%s": null' "$num"; else printf '  "%s": %s' "$num" "$offset"; fi
        done < <(cmd_clock_check)
        echo; echo "}"
    } > "$out/clock_offsets.json"
    log "collected into $out (plot: uv run python scripts/plot_run.py $out)"
}

cmd_monitor() {
    current_run_id
    mkdir -p "logs/$RUN_ID"
    local pidfile="logs/$RUN_ID/monitor.pid" logfile="logs/$RUN_ID/monitor.log" pid
    pid="$( [[ -f "$pidfile" ]] && cat "$pidfile" || true )"
    case "$SUBCOMMAND" in
        start)
            if pid_alive "$pid"; then log "monitor already running (pid=$pid)"; return 0; fi
            local args=(scripts/hfl_monitor.py --profile "$PROFILE" --run-id "$RUN_ID"
                        --interval "$MONITOR_INTERVAL" --port "$HFL_MONITOR_PORT")
            [[ -n "$IDENTITY" ]] && args+=(--identity "$IDENTITY")
            if command -v setsid >/dev/null 2>&1; then
                setsid "$PY" "${args[@]}" > "$logfile" 2>&1 < /dev/null &
            else
                "$PY" "${args[@]}" > "$logfile" 2>&1 < /dev/null &
            fi
            echo $! > "$pidfile"
            sleep 2
            pid_alive "$(cat "$pidfile")" || { tail -n 20 "$logfile" >&2; die "monitor failed to start"; }
            log "monitor started (pid=$(cat "$pidfile")): http://$(local_ip):$HFL_MONITOR_PORT/  (log: $logfile)"
            ;;
        stop)
            if pid_alive "$pid"; then kill "$pid" && log "monitor stopped"; else log "monitor is not running"; fi
            ;;
        status)
            if pid_alive "$pid"; then log "monitor running (pid=$pid) port=$HFL_MONITOR_PORT"; else log "monitor is not running"; fi
            [[ -f "logs/$RUN_ID/monitor/status.json" ]] && head -c 400 "logs/$RUN_ID/monitor/status.json" && echo
            ;;
        *) die "monitor: unknown subcommand $SUBCOMMAND" ;;
    esac
}

case "$COMMAND" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    start-all) cmd_start_all ;;
    stop-all) cmd_stop_all ;;
    resume) cmd_resume ;;
    sampler-stop) cmd_sampler_stop ;;
    status-all) cmd_status_all ;;
    wait) cmd_wait ;;
    collect) cmd_collect ;;
    clock-check) cmd_clock_check | awk -F'\t' '{printf "node %s: offset=%ss rtt=%ss\n", $1, $2, $3}' ;;
    monitor) cmd_monitor ;;
    check) cmd_check ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $COMMAND (see: scripts/hfl.sh help)" ;;
esac
