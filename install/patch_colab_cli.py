# -*- coding: utf-8 -*-
"""Fixes for google-colab-cli 0.7.4. Safe to run more than once.
1. console.py imports termios/tty (POSIX only) at module level, so every `colab` command crashes on Windows even
   though only the interactive console needs them. Wrap the two imports.
2. client.py sends body-less POSTs (e.g. the GPU assign request). Behind some proxies Colab answers those with
   411 Length Required, so the first run never gets a GPU. Send an empty JSON body instead (issue #1, verified:
   no body -> 411, data=b"" -> 411, json={} -> 200).
3. Turns off the update check (it can hang a background agent).
Run with any Python: python patch_colab_cli.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

CLIENT_ANCHOR = "        response = self.session.request(\n            method, endpoint,"
CLIENT_MARK = "h3skill: 411 fix"
CLIENT_FIX = ('        if str(method).upper() == "POST" and "data" not in kwargs and "json" not in kwargs:  # ' + CLIENT_MARK + "\n"
              "            kwargs[\"json\"] = {}\n")


def tool_root():
    try:
        return Path(subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return Path(os.environ.get("APPDATA", Path.home() / ".local" / "share")) / "uv" / "tools"


def patch_console(p):
    s = p.read_text(encoding="utf-8")
    if "termios = None" in s:
        return "已經補過"
    s = s.replace("import termios\n", "try:\n    import termios\nexcept ImportError:  # Windows\n    termios = None\n", 1)
    s = s.replace("import tty\n", "try:\n    import tty\nexcept ImportError:\n    tty = None\n", 1)
    p.write_text(s, encoding="utf-8")
    return "已補丁"


def patch_client(p):
    s = p.read_text(encoding="utf-8")
    if CLIENT_MARK in s:
        return "已經補過"
    if s.count(CLIENT_ANCHOR) != 1:
        return "⚠️ 這個版本的 client.py 跟預期不同，沒有補（411 錯誤若出現，請回報 issue）"
    p.write_text(s.replace(CLIENT_ANCHOR, CLIENT_FIX + CLIENT_ANCHOR, 1), encoding="utf-8")
    return "已補丁"


def main():
    conf = Path.home() / ".config" / "colab-cli"
    conf.mkdir(parents=True, exist_ok=True)
    settings = conf / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    data["enable_update_check"] = False
    settings.write_text(json.dumps(data), encoding="utf-8")
    pkgs = list(tool_root().glob("google-colab-cli/**/colab_cli"))
    if not pkgs:
        sys.exit("找不到 colab_cli：google-colab-cli 裝好了嗎？")
    for pkg in pkgs:
        for name, fn in (("console.py", patch_console), ("client.py", patch_client)):
            if (pkg / name).exists():
                print(f"{fn(pkg / name)}：{pkg / name}")


if __name__ == "__main__":
    main()
