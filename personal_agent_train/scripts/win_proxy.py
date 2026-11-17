"""Windows 侧的本机转发代理:让 WSL 借 Windows 的网络出网。

公司网络(SmartVPN 等)只放行 Windows 进程的出站连接,WSL2 直连外网会超时;mirrored 网络模式下 WSL 能访问 Windows 的
127.0.0.1,所以在 Windows 上起这个代理,WSL 里设 ``http_proxy=https_proxy=http://127.0.0.1:<port>`` 即可。

    python scripts/win_proxy.py --port 18080

只监听 127.0.0.1,不做鉴权、不缓存、不改内容。支持 HTTPS(CONNECT 隧道)与普通 HTTP(apt 等用 http://)。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from urllib.parse import urlsplit

log = logging.getLogger("win_proxy")
BUF = 1 << 16


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(BUF):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def read_head(reader: asyncio.StreamReader) -> bytes:
    return await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)


async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
    try:
        head = await read_head(client_r)
    except Exception:  # noqa: BLE001
        client_w.close()
        return
    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, version = lines[0].split(" ", 2)
    except ValueError:
        client_w.close()
        return
    try:
        if method.upper() == "CONNECT":
            host, _, port = target.rpartition(":")
            up_r, up_w = await asyncio.wait_for(asyncio.open_connection(host.strip("[]"), int(port)), timeout=30)
            client_w.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await client_w.drain()
        else:
            url = urlsplit(target)
            host, port = url.hostname, url.port or 80
            path = (url.path or "/") + (f"?{url.query}" if url.query else "")
            up_r, up_w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=30)
            headers = [l for l in lines[1:] if l and not l.lower().startswith(("proxy-connection:", "proxy-authorization:"))]
            up_w.write(f"{method} {path} {version}\r\n".encode("latin-1") + "\r\n".join(headers).encode("latin-1") + b"\r\n\r\n")
            await up_w.drain()
        log.info("%s %s", method, target)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s %s failed: %s", method, target, exc)
        try:
            client_w.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            await client_w.drain()
        finally:
            client_w.close()
        return
    await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))


async def main(port: int) -> None:
    server = await asyncio.start_server(handle, "127.0.0.1", port, limit=BUF)
    log.warning("proxy listening on 127.0.0.1:%d", port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(asctime)s %(message)s")
    asyncio.run(main(a.port))
