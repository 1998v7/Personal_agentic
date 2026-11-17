"""同步 → 异步桥。

VitaBench 的 agent 是同步方法,并用线程池并发多个任务;执行层是 async。
这里起一个常驻的后台事件循环线程,同步代码把协程提交过去并阻塞等待结果。所有任务共用一个循环,
执行层按 task_id 做并发控制,SQLite 每次事务单开连接,所以多线程提交是安全的。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class AsyncBridge:
    def __init__(self, name: str = "imact-loop") -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()
        self._closed = False

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def run(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        if self._closed:
            raise RuntimeError("AsyncBridge is closed")
        if threading.current_thread() is self._thread:
            raise RuntimeError("AsyncBridge.run() 不能在它自己的事件循环线程里调用(会死锁)")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def call(self, factory: Callable[[], T]) -> T:
        """在循环线程里同步执行一个工厂函数(用于创建必须属于该循环的对象)。"""

        async def _wrap() -> Any:
            return factory()

        return self.run(_wrap())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()
