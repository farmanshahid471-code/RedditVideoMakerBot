#!/usr/bin/env python
"""Pre-flight environment check for Reddit Video Maker Bot.

Run this BEFORE `python main.py`. It fails fast on the mistakes that otherwise
waste 10+ minutes of TTS and scraping before dying at the render step.

    python selfcheck.py

Standard library only on purpose: it has to stay runnable when dependencies
are missing, which is exactly when you need it.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Pure ASCII so this never dies on a cp1252/cp1254 console while trying to
# tell you about a different problem.
OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"

_results = []


def report(status, name, detail=""):
    _results.append((status, name, detail))
    line = f"{status} {name}"
    if detail:
        line += f" - {detail}"
    print(line, flush=True)


def check(name):
    """Decorator: run a function returning (ok, detail) as one report line."""

    def deco(fn):
        def runner():
            try:
                ok, detail = fn()
            except Exception as err:  # a missing dep is a finding, not a crash
                ok, detail = False, f"{type(err).__name__}: {err}"
            report(OK if ok else FAIL, name, detail)

        return runner

    return deco


def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


@check("Python is 3.10-3.12")
def _python_version():
    v = sys.version_info
    allowed = v.major == 3 and v.minor in (10, 11, 12)
    return allowed, (
        f"running {v.major}.{v.minor}.{v.micro} ({sys.executable})"
        + ("" if allowed else " - main.py hard-exits outside 3.10/3.11/3.12")
    )


@check("Dependencies are installed into this venv")
def _venv():
    if not _in_venv():
        return False, f"not in a venv; pip is writing to {sys.base_prefix}"
    return True, f"{Path(sys.prefix).name} -> {sys.prefix}"


@check("Project lives off the system drive")
def _drive():
    root = Path(__file__).resolve().parent
    drive = root.drive or "/"
    free_gb = shutil.disk_usage(str(root)).free / 1e9
    if drive.lower() == "c:":
        return False, f"{root} is on C:; needs ~10 GB free for renders"
    return free_gb > 4, f"{root} | {free_gb:.1f} GB free"


@check("ffmpeg is on PATH and can encode libx264")
def _ffmpeg():
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found on PATH (winget install -e --id Gyan.FFmpeg)"
    found = False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        found = "libx264" in out
    except Exception:
        pass
    return found, "libx264 available" if found else "this ffmpeg build has no libx264"


def _import_and_version(pkg, expect=None):
    try:
        mod = __import__(pkg)
    except ImportError as err:
        return False, f"cannot import {pkg}: {err}"
    got = getattr(mod, "__version__", None)
    if expect and got and not got.startswith(expect):
        return True, f"{pkg} {got} (requirements.txt pins {expect})"
    return True, f"{pkg} {got or '?'}"


# Module -> pinned version in requirements.txt. These are imported at module
# load by main.py's import graph, so a missing one aborts before any work.
REQUIRED = {
    "praw": "7.8.1",
    "moviepy": "2.2.1",
    "ffmpeg": "0.2.0",
    "gtts": "2.5.4",
    "playwright": "1.49.1",
    "requests": "2.32.3",
    "rich": "13.9.4",
    "toml": "0.10.2",
    "translators": "5.9.9",
    "yt_dlp": "2025.10.22",
    "boto3": "1.36.8",
    "elevenlabs": "1.57.0",
    "clean_text": "0.6.0",
    "unidecode": "1.4.0",
    "PIL": None,
    "numpy": None,
}

# Imported unconditionally by reddit/subreddit.py even when ai_similarity is off.
HEAVY = {
    "torch": "2.7.0",
    "transformers": "4.52.4",
    "spacy": "3.8.7",
    "pyttsx3": "2.98",
}


@check("Core packages import")
def _core():
    bad = [f"{p} ({d})" for p, v in REQUIRED.items() for ok, d in [_import_and_version(p, v)] if not ok]
    if not bad:
        return True, f"all {len(REQUIRED)} present"
    shown = "; ".join(bad[:3])
    extra = f"; +{len(bad) - 1} more" if len(bad) > 1 else ""
    return False, f"{shown}{extra}"


@check("Heavy packages import (torch/transformers/spacy)")
def _heavy():
    bad = [f"{p} ({d})" for p, v in HEAVY.items() for ok, d in [_import_and_version(p, v)] if not ok]
    if not bad:
        return True, "importable without GPU"
    return False, "; ".join(bad[:2]) + (f"; +{len(bad) - 1} more" if len(bad) > 1 else "")


@check("spacy en_core_web_sm model present")
def _spacy_model():
    try:
        import spacy
    except ImportError as err:
        return False, f"spacy missing: {err}"
    try:
        spacy.load("en_core_web_sm")
    except Exception:
        return False, "run: python -m spacy download en_core_web_sm"
    return True, "loads"


@check("Playwright Chromium is installed where it is found")
def _browser():
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as err:
        return False, f"playwright missing: {err}"
    try:
        with sync_playwright() as p:
            exe = Path(p.chromium.executable_path)
            if not exe.exists():
                return False, "run: python -m playwright install chromium"
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception as err:
        return False, f"{type(err).__name__}: {str(err).splitlines()[0]}"
    where = f"{exe.parent} (PLAYWRIGHT_BROWSERS_PATH={override})" if override else str(exe.parent)
    return True, f"launches; {where}"


@check("Reddit is reachable from this network")
def _reddit():
    try:
        import requests
    except ImportError as err:
        return False, f"requests missing: {err}"
    try:
        r = requests.get(
            "https://www.reddit.com/",
            headers={"User-Agent": "selfcheck:1.0 (diagnostic)"},
            timeout=20,
        )
    except Exception as err:
        return False, f"no connectivity: {err}"
    if r.status_code == 403:
        return False, "Reddit returned 403 - this IP is blocked (VPN/datacenter/college NAT)"
    return r.ok, f"HTTP {r.status_code}"


@check("config.toml has Reddit credentials")
def _config():
    cfg = Path(__file__).resolve().parent / "config.toml"
    if not cfg.exists():
        return False, "absent - `python main.py` will interview you and create it"
    try:
        import toml

        data = toml.loads(cfg.read_text(encoding="utf-8"))
    except Exception as err:
        return False, f"unreadable: {err}"
    creds = data.get("reddit", {}).get("creds", {})
    missing = [k for k in ("client_id", "client_secret", "username", "password") if not creds.get(k)]
    if missing:
        return False, f"empty keys: {', '.join(missing)}"
    voice = data.get("settings", {}).get("tts", {}).get("voice_choice", "?")
    needs_key = {"elevenlabs": "elevenlabs_api_key", "OpenAI": "openai_api_key"}.get(str(voice))
    if needs_key and not data.get("settings", {}).get("tts", {}).get(needs_key):
        return False, f"voice_choice={voice} but {needs_key} is empty"
    if str(voice) == "tiktok" and not data.get("settings", {}).get("tts", {}).get("tiktok_sessionid"):
        return False, "voice_choice=tiktok needs a sessionid; googletranslate needs nothing"
    return True, f"creds set, voice_choice={voice}"


def _warn(name, detail):
    report(WARN, name, detail)


def main() -> int:
    print(f"\nReddit Video Maker Bot - pre-flight check\n{'-' * 56}")
    print(f"python  : {sys.version.split()[0]}  {sys.executable}")
    print(f"cwd     : {Path.cwd()}")
    print(f"encoder : RVM_VIDEO_ENCODER={os.environ.get('RVM_VIDEO_ENCODER', 'libx264 (default)')}")
    for var in ("PLAYWRIGHT_BROWSERS_PATH", "TEMP", "HF_HOME"):
        val = os.environ.get(var)
        if val:
            print(f"{var:<27}: {val}")
    print("-" * 56)

    for fn in (
        _python_version,
        _venv,
        _drive,
        _ffmpeg,
        _core,
        _heavy,
        _spacy_model,
        _browser,
        _reddit,
        _config,
    ):
        fn()

    if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        _warn(
            "PLAYWRIGHT_BROWSERS_PATH unset",
            "browsers default to your C: profile; setx it if C is tight",
        )

    fails = [r for r in _results if r[0] == FAIL]
    print("-" * 56)
    if fails:
        print(f"{len(fails)} problem(s). Fix the first one, then re-run this script:")
        for _, name, detail in fails[:1]:
            print(f"  -> {name}: {detail}")
        return 1
    print("All checks passed. Next: `python main.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
