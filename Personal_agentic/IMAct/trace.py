"""逐步落盘的诊断轨迹。只记录业务输入输出,不记录 HTTP 鉴权 header。"""

from __future__ import annotations

import contextvars
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_context = contextvars.ContextVar("imact_trace_context", default={})


@contextmanager
def trace_scope(**values):
    token = _context.set({**_context.get(), **values})
    try:
        yield
    finally:
        _context.reset(token)


class TraceRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.seq = 0
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                self.seq = sum(1 for _ in stream)

    def emit(self, stage: str, **data):
        with self.lock:
            self.seq += 1
            record = {"seq": self.seq, "time": datetime.now(timezone.utc).isoformat(),
                      "stage": stage, "correlation": dict(_context.get()), "data": data}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
