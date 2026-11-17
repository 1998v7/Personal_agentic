"""下载 PersonaMem-v3 的评测数据(HuggingFace ``bowen-upenn/PersonaMem-v3`` 的 ``backend/{uid}/``)到 ``data/personamem_v3/backend/``。

    python -X utf8 -m IMAct.benchmarks.personamem_v3.download                 # 全部 99 个 persona(约 1.1 GB)
    python -X utf8 -m IMAct.benchmarks.personamem_v3.download --users 25 36   # 只下这几个

每个 persona 下载评测要用的 8 个文件:五个 app 的历史、calendar.json、profile.json(评分与部分 prompt 用)、test.json(题目)。
``persona.html``(人工浏览页,约 4 MB/人)默认不下载。已存在且大小一致的文件跳过,可以随时中断后重跑。
数据许可证是 CC-BY-NC-4.0(非商用)。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

from .data import DATA_DIR, EVAL_FILES

REPO = "bowen-upenn/PersonaMem-v3"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def list_remote(client: httpx.Client) -> dict[str, int]:
    """``backend/{uid}/{file}`` → 字节数。"""
    resp = client.get(API, timeout=60)
    resp.raise_for_status()
    out = {}
    for item in resp.json():
        if item.get("type") == "file" and item["path"].startswith("backend/"):
            out[item["path"]] = int((item.get("lfs") or {}).get("size") or item.get("size") or 0)
    return out


def download(users: list[str] | None, dest: Path, *, include_html: bool = False, retries: int = 3) -> int:
    wanted_files = set(EVAL_FILES) | ({"persona.html"} if include_html else set())
    with httpx.Client(follow_redirects=True, headers={"User-Agent": "imact"}) as client:
        remote = list_remote(client)
        all_users = sorted({p.split("/")[1] for p in remote}, key=lambda u: (len(u), u))
        if users:
            missing = [u for u in users if u not in all_users]
            if missing:
                print(f"数据集里没有这些 persona: {', '.join(missing)}", file=sys.stderr)
                return 2
            all_users = [u for u in all_users if u in set(users)]
        jobs = [(p, size) for p, size in sorted(remote.items()) if p.split("/")[1] in all_users and p.split("/")[-1] in wanted_files]
        total = sum(size for _, size in jobs)
        print(f"{len(all_users)} 个 persona, {len(jobs)} 个文件, 共 {total / 1e6:.0f} MB -> {dest}")
        done = 0
        for i, (path, size) in enumerate(jobs, 1):
            target = dest / path.removeprefix("backend/")
            if target.is_file() and target.stat().st_size == size:
                done += size
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            for attempt in range(retries):
                try:
                    with client.stream("GET", FILE_URL.format(path=path), timeout=120) as resp:
                        resp.raise_for_status()
                        with open(tmp, "wb") as f:
                            for chunk in resp.iter_bytes(1 << 20):
                                f.write(chunk)
                    if tmp.stat().st_size != size:
                        raise IOError(f"大小不符: {tmp.stat().st_size} != {size}")
                    tmp.replace(target)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt + 1 == retries:
                        print(f"下载失败 {path}: {exc}", file=sys.stderr)
                        return 1
                    time.sleep(2 * (attempt + 1))
            done += size
            print(f"  [{i}/{len(jobs)}] {path}  ({done / 1e6:.0f}/{total / 1e6:.0f} MB)", flush=True)
    print("完成")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="下载 PersonaMem-v3 评测数据(HuggingFace)")
    p.add_argument("--users", nargs="*", default=None, help="只下载这些 persona(目录名,如 25);缺省全部")
    p.add_argument("--dest", default=str(DATA_DIR / "backend"), help="目标目录,缺省 data/personamem_v3/backend")
    p.add_argument("--include-html", action="store_true", help="同时下载 persona.html(人工浏览页,约 4 MB/人,评测不需要)")
    args = p.parse_args(argv)
    return download(args.users, Path(args.dest), include_html=args.include_html)


if __name__ == "__main__":
    raise SystemExit(main())
