"""
mj_viewer.py — watch_* 腳本共用的 MuJoCo 即時視窗工具

  ensure_mjpython()      macOS 的 mujoco.viewer.launch_passive 必須在 mjpython 底下跑；
                         用 python 啟動時，改用同一個 conda 環境的 mjpython 重新執行目前的腳本（參數不變）
  set_overlay(viewer, …) 在視窗左上角疊加一欄「名稱 / 數值」的資訊面板
"""
import os
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

_REEXEC_ENV = "MJ_VIEWER_REEXEC"


def ensure_mjpython():
    if sys.platform != "darwin" or isinstance(mujoco.viewer._MJPYTHON, mujoco.viewer._MjPythonBase):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists() or os.environ.get(_REEXEC_ENV):
        sys.exit(f"macOS 的 MuJoCo 視窗需要 mjpython，請改用：mjpython {' '.join(sys.argv)}\n"
                 "或加 --mode video 錄成影片")
    script = str(Path(sys.argv[0]).resolve())
    print(f"↪ 改用 {mjpython} 重新啟動（macOS 的 MuJoCo 視窗需要 mjpython）")
    os.environ[_REEXEC_ENV] = "1"
    os.execv(str(mjpython), [str(mjpython), script, *sys.argv[1:]])


def set_overlay(viewer, rows):
    """rows：[(名稱, 數值字串), …]，顯示在視窗左上角。"""
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      "\n".join(k for k, _ in rows), "\n".join(v for _, v in rows)))
