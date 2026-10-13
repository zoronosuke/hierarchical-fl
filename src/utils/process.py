"""エントリーポイント用のプロセス終了処理。"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable


def run_main(main: Callable[[], None]) -> None:
    """main() を実行し、例外で終わったら必ずプロセスを終了させる。

    Flower の start_server / start_client は、例外で抜けると gRPC のスレッドが残り、
    プロセスがポートを開いたまま終了しない (親子の連鎖停止も止まる)。
    そのため例外時は traceback を出してから os._exit(1) で確実に終了する。
    正常終了時は通常どおり終了する。
    """
    try:
        main()
    except KeyboardInterrupt:
        _flush()
        os._exit(130)
    except BaseException:  # noqa: BLE001 - 終了コードを確実に返すため
        traceback.print_exc()
        _flush()
        os._exit(1)


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
