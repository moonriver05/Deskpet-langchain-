"""Local RSSHub Docker service helpers.

The desktop pet only consumes RSSHub over HTTP.  This module optionally keeps a
local Docker RSSHub instance running and points RSSHub-backed feeds at it.
"""

import os
import subprocess
import threading
import hashlib
import json
from urllib.parse import urlparse, urlunparse

from pet_core.config import app_config
from pet_core.learning_logger import LEARNING_DATA_DIR


KNOWN_PUBLIC_RSSHUB_HOSTS = {
    "rsshub.app",
    "rsshub.rssforever.com",
    "rsshub.uneasy.win",
}
RSSHUB_STATE_PATH = os.path.join(LEARNING_DATA_DIR, "rss_content", "local_rsshub_state.json")


def _truthy(value, default=False):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _int_config(key, default, min_value=1, max_value=65535):
    try:
        value = int(app_config.get(key, default))
    except Exception:
        value = int(default)
    return max(min_value, min(max_value, value))


def local_rsshub_enabled():
    return _truthy(app_config.get("rsshub_local.enabled", "false"), default=False)


def local_rsshub_base_url():
    configured = str(app_config.get("rsshub_local.base_url", "") or "").strip().rstrip("/")
    if configured:
        return configured
    port = _int_config("rsshub_local.host_port", 1200)
    return f"http://127.0.0.1:{port}"


def configure_local_rsshub_base():
    """Point RSSHub-based routes to local Docker when enabled.

    Returns the active base URL, or an empty string if local RSSHub is disabled.
    """
    if not local_rsshub_enabled():
        return ""
    if not _truthy(app_config.get("rsshub_local.use_as_base_url", "true"), default=True):
        return ""
    base_url = local_rsshub_base_url()
    app_config.set("rss_recommender.base_url", base_url)
    _rewrite_saved_rsshub_sources(base_url)
    return base_url


def _rewrite_saved_rsshub_sources(base_url):
    """Move existing public RSSHub source URLs to the configured local base.

    User-added sources often store a full public mirror URL.  If local RSSHub is
    enabled, preserving those full mirror URLs would keep hitting the old mirror.
    Only known RSSHub hosts are rewritten; direct feeds like GitHub/arXiv stay as
    they are.
    """
    try:
        from pet_core.rss_content import list_sources, save_sources
    except Exception as e:
        print(f"[RSSHub] source rewrite skipped: {e}")
        return

    parsed_base = urlparse(base_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        return

    changed = False
    sources = list_sources()
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = str(source.get("feed_url") or "").strip()
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host not in KNOWN_PUBLIC_RSSHUB_HOSTS:
            continue
        new_url = urlunparse(
            (
                parsed_base.scheme,
                parsed_base.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
        if new_url != url:
            source["feed_url"] = new_url
            changed = True
    if changed:
        save_sources(sources)
        print(f"[RSSHub] 已把公共 RSSHub 源切换到本地：{base_url}")


def _docker_hidden_flags():
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run_docker(args, timeout=12):
    return subprocess.run(
        ["docker"] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_docker_hidden_flags(),
    )


def _docker_env_args():
    env = _docker_env()
    args = []
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    return args


def _docker_env():
    env = {
        "NODE_ENV": "production",
        "CACHE_TYPE": str(app_config.get("rsshub_local.cache_type", "memory") or "memory").strip() or "memory",
    }

    pixiv_token = str(app_config.get("rsshub_local.pixiv_refresh_token", "") or "").strip()
    if pixiv_token:
        env["PIXIV_REFRESHTOKEN"] = pixiv_token

    bilibili_cookie = (
        str(app_config.get("rsshub_local.bilibili_cookie", "") or "").strip()
        or str(app_config.get("rss_recommender.bilibili_cookie", "") or "").strip()
    )
    if bilibili_cookie:
        env["BILIBILI_COOKIE"] = bilibili_cookie

    extra = str(app_config.get("rsshub_local.extra_env", "") or "")
    for line in extra.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value.strip()

    return env


def _env_hash():
    env = _docker_env()
    payload = json.dumps(env, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _env_has_credentials():
    env = _docker_env()
    return any(key not in {"NODE_ENV", "CACHE_TYPE"} and str(value).strip() for key, value in env.items())


def _read_state():
    try:
        with open(RSSHUB_STATE_PATH, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_state(obj):
    try:
        os.makedirs(os.path.dirname(RSSHUB_STATE_PATH), exist_ok=True)
        tmp = RSSHUB_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, RSSHUB_STATE_PATH)
    except Exception as e:
        print(f"[RSSHub] 写入本地容器状态失败：{e}")


def ensure_local_rsshub():
    """Ensure the configured Docker RSSHub container exists and is running."""
    if not local_rsshub_enabled():
        return {"ok": True, "reason": "disabled"}
    if not _truthy(app_config.get("rsshub_local.auto_start", "true"), default=True):
        return {"ok": True, "reason": "auto_start_disabled"}

    name = str(app_config.get("rsshub_local.container_name", "rsshub") or "rsshub").strip()
    image = str(app_config.get("rsshub_local.image", "diygod/rsshub") or "diygod/rsshub").strip()
    port = _int_config("rsshub_local.host_port", 1200)
    env_hash = _env_hash()

    try:
        running = _run_docker(["inspect", "-f", "{{.State.Running}}", name], timeout=8)
    except FileNotFoundError:
        print("[RSSHub] Docker 未安装或 docker 命令不可用，已跳过本地 RSSHub 自动启动。")
        return {"ok": False, "reason": "docker_not_found"}
    except subprocess.TimeoutExpired:
        print("[RSSHub] docker inspect 超时，已跳过本地 RSSHub 自动启动。")
        return {"ok": False, "reason": "docker_timeout"}
    except Exception as e:
        print(f"[RSSHub] docker inspect 失败：{e}")
        return {"ok": False, "reason": str(e)}

    if running.returncode == 0:
        state = _read_state()
        same_container = (
            state.get("container_name") == name
            and state.get("image") == image
            and int(state.get("host_port") or port) == port
        )
        env_changed = same_container and state.get("env_hash") and state.get("env_hash") != env_hash
        untracked_with_credentials = (not state.get("container_name")) and _env_has_credentials()
        if (env_changed or untracked_with_credentials) and _truthy(
            app_config.get("rsshub_local.recreate_on_env_change", "true"),
            default=True,
        ):
            removed = _run_docker(["rm", "-f", name], timeout=20)
            if removed.returncode == 0:
                print(f"[RSSHub] RSSHub 凭证/环境变量变化，已移除旧容器并准备重建：{name}")
                running = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
            else:
                print(f"[RSSHub] 凭证变化但重建容器失败：{removed.stderr.strip() or removed.stdout.strip()}")

    if running.returncode == 0 and running.stdout.strip().lower() == "true":
        _write_state({"container_name": name, "image": image, "host_port": port, "env_hash": env_hash})
        print(f"[RSSHub] 本地容器已运行：{name} -> {local_rsshub_base_url()}")
        return {"ok": True, "reason": "already_running"}

    if running.returncode == 0:
        start = _run_docker(["start", name], timeout=20)
        if start.returncode == 0:
            _write_state({"container_name": name, "image": image, "host_port": port, "env_hash": env_hash})
            print(f"[RSSHub] 已启动本地容器：{name} -> {local_rsshub_base_url()}")
            return {"ok": True, "reason": "started"}
        print(f"[RSSHub] docker start 失败：{start.stderr.strip() or start.stdout.strip()}")
        return {"ok": False, "reason": "start_failed"}

    run_args = [
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{port}:1200",
    ]
    run_args.extend(_docker_env_args())
    run_args.append(image)

    try:
        created = _run_docker(run_args, timeout=90)
    except subprocess.TimeoutExpired:
        print("[RSSHub] docker run 超时；如果是第一次拉镜像，稍后可再启动一次桌宠。")
        return {"ok": False, "reason": "run_timeout"}
    if created.returncode == 0:
        _write_state({"container_name": name, "image": image, "host_port": port, "env_hash": env_hash})
        print(f"[RSSHub] 已创建并启动本地容器：{name} -> {local_rsshub_base_url()}")
        return {"ok": True, "reason": "created"}

    err = created.stderr.strip() or created.stdout.strip()
    print(f"[RSSHub] docker run 失败：{err}")
    return {"ok": False, "reason": "run_failed"}


def start_local_rsshub_background():
    configure_local_rsshub_base()
    if not local_rsshub_enabled():
        return None
    thread = threading.Thread(target=ensure_local_rsshub, name="LocalRSSHubStarter", daemon=True)
    thread.start()
    return thread
