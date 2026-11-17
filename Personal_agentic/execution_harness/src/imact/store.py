"""SQLite 工作区：先记录输入；模型在事务外运行；状态、笔记与动作原子提交。"""

import json
import sqlite3
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from .errors import HarnessError


def dumps(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("Use a temporary file; each transaction opens its own connection")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, rev INTEGER NOT NULL, workspace TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    task_id TEXT NOT NULL, event_id TEXT NOT NULL, base_rev INTEGER NOT NULL,
                    input_hash TEXT NOT NULL, input TEXT NOT NULL, status TEXT NOT NULL,
                    output TEXT, error TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    PRIMARY KEY (task_id, event_id));
                CREATE TABLE IF NOT EXISTS notes (
                    task_id TEXT NOT NULL, note_key TEXT NOT NULL, value TEXT NOT NULL,
                    PRIMARY KEY (task_id, note_key));
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    event_id TEXT NOT NULL, context TEXT, decision TEXT, error TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
            """)
            # 兼容已有 SQLite 文件，不丢弃旧轨迹。
            for table, column in (("events", "staged_workspace"), ("attempts", "metadata")):
                if column not in {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def claim(self, request: dict) -> tuple[dict | None, dict | None]:
        """返回 (workspace, cached_output)。跨进程只允许同任务一个在途事件。"""
        tid, eid = request["task_id"], request["event_id"]
        raw = dumps(request)
        fingerprint = sha256(raw.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event = db.execute(
                "SELECT * FROM events WHERE task_id=? AND event_id=?", (tid, eid)
            ).fetchone()
            if event:
                if event["input_hash"] != fingerprint:
                    raise HarnessError(
                        "event_payload_conflict", "同一 event_id 的输入发生变化", 409
                    )
                if event["status"] == "committed":
                    return None, json.loads(event["output"])
                if event["status"] == "processing":
                    raise HarnessError(
                        "event_in_progress", "该事件正在处理，请稍后以相同请求重试", 409
                    )
                if event["status"] == "rejected":
                    raise HarnessError(
                        "event_rejected", "此输入此前已被拒绝，修正后请使用新 event_id", 409
                    )
                if event["status"] == "superseded":
                    raise HarnessError(
                        "event_superseded", "事件已被后续用户输入取代，不可重放", 409
                    )
            task = db.execute("SELECT * FROM tasks WHERE task_id=?", (tid,)).fetchone()
            rev = task["rev"] if task else 0
            if request["base_rev"] != rev:
                raise HarnessError("revision_conflict", f"当前 rev={rev}", 409)
            if task and request["event"]["type"] == "start":
                raise HarnessError("task_exists", "新任务请使用新的 task_id", 409)
            others = db.execute(
                "SELECT * FROM events WHERE task_id=? AND event_id<>? AND status IN ('failed','processing') ORDER BY rowid DESC",
                (tid, eid),
            ).fetchall()
            corrective = request["event"]["type"] in {"user_message", "terminate", "resume"}
            if any(r["status"] == "processing" for r in others) or (others and not corrective):
                raise HarnessError("pending_event", "请先完成此任务之前未提交的事件", 409)
            staged = event["staged_workspace"] if event else None
            staged_eid = eid
            if staged is None and others:
                latest = next((r for r in others if r["staged_workspace"]), None)
                if latest:
                    staged, staged_eid = latest["staged_workspace"], latest["event_id"]
            if not task and request["event"]["type"] != "start" and staged is None:
                raise HarnessError("task_not_found", "请先投递 start", 404)
            if event:
                db.execute(
                    "UPDATE events SET status='processing',error=NULL WHERE task_id=? AND event_id=?",
                    (tid, eid),
                )
            else:
                db.execute(
                    "INSERT INTO events(task_id,event_id,base_rev,input_hash,input,status) VALUES(?,?,?,?,?,'processing')",
                    (tid, eid, rev, fingerprint, raw),
                )
            workspace = json.loads(task["workspace"]) if task else None
            if staged is not None:
                workspace = json.loads(staged)
                workspace["_observed_event"] = staged_eid
            elif workspace is not None:
                workspace["notes"] = {
                    row["note_key"]: json.loads(row["value"])
                    for row in db.execute(
                        "SELECT note_key,value FROM notes WHERE task_id=?", (tid,)
                    )
                }
            if workspace is not None and "task_origin" not in workspace:
                origin = db.execute(
                    "SELECT event_id FROM events WHERE task_id=? AND status='committed' ORDER BY rowid LIMIT 1",
                    (tid,),
                ).fetchone()
                if origin:
                    workspace["task_origin"] = origin["event_id"]
            return workspace, None

    def stage_observation(self, tid: str, eid: str, workspace: dict):
        """只暂存已接收事实，不提交未经校验的模型修改。"""
        with self.connect() as db:
            db.execute(
                "UPDATE events SET staged_workspace=? WHERE task_id=? AND event_id=?",
                (dumps(workspace), tid, eid),
            )

    def repair_feedback(self, tid: str, eid: str) -> list[dict]:
        with self.connect() as db:
            return [
                {
                    "attempt": r["id"],
                    "error": json.loads(r["error"]),
                    "previous_decision": (r["decision"] or "")[:12000],
                    "decision_truncated": len(r["decision"] or "") > 12000,
                }
                for r in db.execute(
                    "SELECT id,error,decision FROM attempts WHERE task_id=? AND event_id=? AND error IS NOT NULL ORDER BY id",
                    (tid, eid),
                )
            ]

    def begin_attempt(self, tid: str, eid: str, context: dict, metadata: dict | None = None) -> int:
        with self.connect() as db:
            return db.execute(
                "INSERT INTO attempts(task_id,event_id,context,metadata) VALUES(?,?,?,?)",
                (tid, eid, dumps(context), dumps(metadata or {})),
            ).lastrowid

    def record_decision(self, attempt_id: int, decision: Any, metadata: dict | None = None):
        with self.connect() as db:
            previous = db.execute(
                "SELECT metadata FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            merged = {**json.loads(previous["metadata"] or "{}"), **(metadata or {})}
            db.execute(
                "UPDATE attempts SET decision=?,metadata=? WHERE id=?",
                (dumps(decision), dumps(merged), attempt_id),
            )

    def fail(
        self,
        tid: str,
        eid: str,
        error: dict,
        attempt_id: int | None,
        rejected=False,
        keep_processing=False,
    ):
        with self.connect() as db:
            db.execute(
                "UPDATE events SET status=?,error=? WHERE task_id=? AND event_id=? AND status='processing'",
                (
                    "rejected" if rejected else ("processing" if keep_processing else "failed"),
                    dumps(error),
                    tid,
                    eid,
                ),
            )
            if attempt_id is not None:
                db.execute("UPDATE attempts SET error=? WHERE id=?", (dumps(error), attempt_id))

    def commit(self, request: dict, workspace: dict, output: dict, failed=False):
        tid, eid, base = request["task_id"], request["event_id"], request["base_rev"]
        data = {k: v for k, v in workspace.items() if k != "notes"}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT rev FROM tasks WHERE task_id=?", (tid,)).fetchone()
            if (row["rev"] if row else 0) != base:
                raise HarnessError("revision_conflict", "提交时版本变化", 409)
            if row:
                db.execute(
                    "UPDATE tasks SET rev=?,workspace=? WHERE task_id=?",
                    (base + 1, dumps(data), tid),
                )
            else:
                db.execute("INSERT INTO tasks VALUES(?,?,?)", (tid, base + 1, dumps(data)))
            for key, note in workspace["notes"].items():
                db.execute(
                    "INSERT INTO notes VALUES(?,?,?) ON CONFLICT(task_id,note_key) DO UPDATE SET value=excluded.value",
                    (tid, key, dumps(note)),
                )
            changed = db.execute(
                "UPDATE events SET status='committed',output=?,staged_workspace=NULL WHERE task_id=? AND event_id=? AND status=?",
                (dumps(output), tid, eid, "failed" if failed else "processing"),
            ).rowcount
            if changed != 1:
                raise HarnessError("event_not_owned", "事件不再处于处理中", 409)
            db.execute(
                "UPDATE events SET status='superseded',staged_workspace=NULL WHERE task_id=? AND event_id<>? AND base_rev=? AND status='failed'",
                (tid, eid, base),
            )

    def inspect(self, tid: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (tid,)).fetchone()
            if not row:
                raise HarnessError("task_not_found", "任务尚无已提交状态", 404)
            value = json.loads(row["workspace"])
            value["notes"] = [
                json.loads(r["value"])
                for r in db.execute(
                    "SELECT value FROM notes WHERE task_id=? ORDER BY note_key", (tid,)
                )
            ]
            return {"task_id": tid, "rev": row["rev"], "workspace": value}

    def trace(self, tid: str) -> dict:
        with self.connect() as db:
            events = [
                dict(r)
                for r in db.execute("SELECT * FROM events WHERE task_id=? ORDER BY rowid", (tid,))
            ]
            attempts = [
                dict(r)
                for r in db.execute("SELECT * FROM attempts WHERE task_id=? ORDER BY id", (tid,))
            ]
        for row in events + attempts:
            for key in (
                "input",
                "output",
                "error",
                "context",
                "decision",
                "metadata",
                "staged_workspace",
            ):
                if row.get(key) is not None:
                    row[key] = json.loads(row[key])
        return {"task_id": tid, "events": events, "attempts": attempts}

    def evidence(self, tid: str, reference: str) -> dict:
        """按工作区 source_refs 取原始证据；不读取任意路径或跨任务材料。"""
        with self.connect() as db:
            row = db.execute(
                "SELECT input,status FROM events WHERE task_id=? AND event_id=?", (tid, reference)
            ).fetchone()
            if row:
                return {
                    "reference": reference,
                    "event_status": row["status"],
                    "event": json.loads(row["input"])["event"],
                }
            eid, separator, cid = reference.rpartition("/")
            if separator:
                row = db.execute(
                    "SELECT input,status FROM events WHERE task_id=? AND event_id=?", (tid, eid)
                ).fetchone()
                if row:
                    event = json.loads(row["input"])["event"]
                    for result in event.get("results", []):
                        if result["call_id"] == cid:
                            return {
                                "reference": reference,
                                "event_status": row["status"],
                                "result": result,
                            }
        raise HarnessError("evidence_not_found", "本任务没有该证据引用", 404)

    def recover_interrupted(self, tid: str, eid: str):
        """仅在确认旧进程已停止后由 CLI 显式调用，不自动抢占租约。"""
        with self.connect() as db:
            n = db.execute(
                "UPDATE events SET status='failed',error=? WHERE task_id=? AND event_id=? AND status='processing'",
                (dumps({"code": "interrupted", "message": "operator recovery"}), tid, eid),
            ).rowcount
            if n != 1:
                raise HarnessError("not_processing", "没有可恢复的 processing 事件", 409)
