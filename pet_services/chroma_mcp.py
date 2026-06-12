"""Persistent Chroma MCP client service.

This module owns the long-lived MCP worker thread and exposes synchronous
helpers for pet.py. It deliberately has no dependency on PyQt or MySQL.
"""

import asyncio
import atexit
import concurrent.futures
import json
import os
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


CHROMA_COLLECTION_KB = "pet_knowledge_base"
CHROMA_COLLECTION_MEM = "pet_user_memory"

CHROMA_MCP_CONFIG = {
    "container_name": os.environ.get("CHROMA_MCP_CONTAINER") or "chroma-mcp",
    "container_command": [
        "chroma-mcp", "--client-type", "persistent", "--data-dir", "/chroma_data",
    ],
    "fallback_to_run": True,
    "fallback_image": "mcp/chroma",
    "fallback_volume_name": "pet_desktop_chroma",
    "fallback_data_dir_in_container": "/chroma_data",
}


def configure_chroma(container_name=None):
    if container_name:
        CHROMA_MCP_CONFIG["container_name"] = container_name


def _container_is_running(name):
    if not name:
        return False
    import subprocess
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        print("[Chroma MCP] 未找到 docker 可执行文件。")
        return False
    except Exception as e:
        print(f"[Chroma MCP] docker inspect 调用失败: {e}")
        return False
    if out.returncode != 0:
        return False
    return out.stdout.strip().lower() == "true"


def _chroma_docker_stdio_params():
    name = CHROMA_MCP_CONFIG.get("container_name") or "chroma-mcp"
    cmd = list(CHROMA_MCP_CONFIG.get("container_command") or ["chroma-mcp"])
    if _container_is_running(name):
        print(f"[Chroma MCP] 复用已运行容器 `{name}`（docker exec -i {name} {' '.join(cmd)}）。")
        return StdioServerParameters(command="docker", args=["exec", "-i", name] + cmd)
    if not CHROMA_MCP_CONFIG.get("fallback_to_run"):
        raise RuntimeError(
            f"容器 `{name}` 未运行，且已禁用 docker run 回退。请先 `docker start {name}` 再启动桌宠。"
        )
    image = CHROMA_MCP_CONFIG.get("fallback_image", "mcp/chroma")
    vol = CHROMA_MCP_CONFIG.get("fallback_volume_name", "pet_desktop_chroma")
    data_dir = CHROMA_MCP_CONFIG.get("fallback_data_dir_in_container", "/chroma_data")
    args = [
        "run", "-i", "--rm",
        "-v", f"{vol}:{data_dir}",
        "--entrypoint", "chroma-mcp",
        image,
        "--client-type", "persistent",
        "--data-dir", data_dir,
    ]
    print(f"[Chroma MCP] 容器 `{name}` 未运行，回退到一次性容器 `docker run --rm {image}`（持久化卷 {vol}）。")
    return StdioServerParameters(command="docker", args=args)


def _chrom_tool_first_text(result):
    if not result or not getattr(result, "content", None):
        return None
    block = result.content[0]
    return getattr(block, "text", None) if block is not None else None


def _chrom_tool_to_dict(result):
    if result is None:
        return None
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict) and sc:
        return sc
    text = _chrom_tool_first_text(result)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


_chrom_worker_thread = None
_chrom_worker_loop = None
_chrom_request_q = None
_chrom_worker_lock = threading.Lock()
_chrom_ready = threading.Event()
_CHROM_SHUTDOWN = object()


def _chrom_worker_main():
    global _chrom_worker_loop, _chrom_request_q
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    request_q = asyncio.Queue()
    with _chrom_worker_lock:
        _chrom_worker_loop = loop
        _chrom_request_q = request_q

    async def runner():
        params = _chroma_docker_stdio_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _chrom_ready.set()
                while True:
                    tool_name, arguments, py_fut = await request_q.get()
                    if tool_name is _CHROM_SHUTDOWN:
                        py_fut.set_result(None)
                        break
                    try:
                        r = await session.call_tool(tool_name, arguments)
                        py_fut.set_result(r)
                    except BaseException as e:
                        py_fut.set_exception(e)

    try:
        loop.run_until_complete(runner())
    except BaseException as e:
        print("[Chroma MCP] worker stopped:", e)
    finally:
        _chrom_ready.clear()
        with _chrom_worker_lock:
            if _chrom_worker_loop is loop:
                _chrom_worker_loop = None
                _chrom_request_q = None
            if _chrom_worker_thread is threading.current_thread():
                globals()["_chrom_worker_thread"] = None
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _chrom_ensure_worker():
    global _chrom_worker_thread
    with _chrom_worker_lock:
        need_start = _chrom_worker_thread is None or not _chrom_worker_thread.is_alive()
        if need_start:
            _chrom_ready.clear()
            _chrom_worker_thread = threading.Thread(
                target=_chrom_worker_main, name="ChromaMcpWorker", daemon=True
            )
            _chrom_worker_thread.start()
    if not _chrom_ready.wait(timeout=180):
        raise RuntimeError(
            "Chroma MCP worker startup timed out; check Docker and the mcp/chroma image."
        )


def _chrom_run_tool(tool_name, arguments):
    _chrom_ensure_worker()
    py_fut = concurrent.futures.Future()
    with _chrom_worker_lock:
        loop = _chrom_worker_loop
        request_q = _chrom_request_q
    if loop is None or request_q is None:
        raise RuntimeError("Chroma MCP worker is not ready")

    async def enqueue():
        await request_q.put((tool_name, arguments, py_fut))

    asyncio.run_coroutine_threadsafe(enqueue(), loop).result(timeout=60)
    return py_fut.result(timeout=600)


def _chrom_shutdown_worker():
    global _chrom_worker_thread, _chrom_worker_loop, _chrom_request_q
    try:
        with _chrom_worker_lock:
            thread = _chrom_worker_thread
            loop = _chrom_worker_loop
            request_q = _chrom_request_q
        if thread is None or not thread.is_alive():
            return
        if loop is None or request_q is None:
            return
        py_fut = concurrent.futures.Future()

        async def shutdown():
            await request_q.put((_CHROM_SHUTDOWN, None, py_fut))

        asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=30)
        py_fut.result(timeout=15)
    except Exception:
        pass


atexit.register(_chrom_shutdown_worker)


def chrom_distance_to_sim(d):
    try:
        d = float(d)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + max(0.0, d))


def chroma_query_documents_sync(collection_name, query_texts, n_results, where=None, where_document=None):
    args = {
        "collection_name": collection_name,
        "query_texts": query_texts,
        "n_results": int(n_results),
    }
    if where is not None:
        args["where"] = where
    if where_document is not None:
        args["where_document"] = where_document
    r = _chrom_run_tool("chroma_query_documents", args)
    return _chrom_tool_to_dict(r)


def chroma_add_documents_sync(collection_name, documents, ids, metadatas=None):
    args = {"collection_name": collection_name, "documents": documents, "ids": ids}
    if metadatas is not None:
        args["metadatas"] = metadatas
    _chrom_run_tool("chroma_add_documents", args)


def chroma_get_documents_sync(collection_name, ids=None, where=None, where_document=None,
                              include=None, limit=None):
    args = {"collection_name": collection_name}
    if ids is not None:
        args["ids"] = list(ids)
    if where is not None:
        args["where"] = where
    if where_document is not None:
        args["where_document"] = where_document
    if include is not None:
        args["include"] = list(include)
    if limit is not None:
        args["limit"] = int(limit)
    r = _chrom_run_tool("chroma_get_documents", args)
    return _chrom_tool_to_dict(r)


def chroma_delete_documents_sync(collection_name, ids):
    if not ids:
        return
    _chrom_run_tool("chroma_delete_documents", {"collection_name": collection_name, "ids": list(ids)})
