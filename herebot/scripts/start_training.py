#!/usr/bin/env python3
"""
HereBot Training Launcher - creates FIFO IPC, starts HereBot, manages lifecycle.
"""
import os, sys, signal, subprocess, time

FIFO_PATH = "/tmp/herebot_in"
LOG_PATH = "/tmp/herebot_out.log"
PID_PATH = "/home/hrirake/herebot/herebot/curiosity_state/training.pid"
REPO_ROOT = "/home/hrirake/herebot"
_bot_proc = None

def cleanup(signum=None, frame=None):
    global _bot_proc
    if _bot_proc and _bot_proc.poll() is None:
        _bot_proc.terminate()
        try:
            _bot_proc.wait(timeout=5)
        except:
            _bot_proc.kill()
    try:
        if os.path.exists(FIFO_PATH):
            os.remove(FIFO_PATH)
    except:
        pass
    sys.exit(0)

def main():
    global _bot_proc

    if os.path.exists(FIFO_PATH):
        os.remove(FIFO_PATH)
    os.mkfifo(FIFO_PATH)
    print(f"[Launcher] FIFO created: {FIFO_PATH}", flush=True)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + "/herebot"

    _bot_proc = subprocess.Popen(
        [sys.executable, "main.py"],
        stdin=open(FIFO_PATH, "r"),
        stdout=open(LOG_PATH, "w"),
        stderr=subprocess.STDOUT,
        cwd=os.path.join(REPO_ROOT, "herebot"),
        env=env,
        preexec_fn=os.setsid,
    )

    pid_info = (
        f"PID={_bot_proc.pid}\n"
        f"START_TIME={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"SESSION_ID=training_{int(time.time())}\n"
        f"REPO={REPO_ROOT}\n"
        f"REPAIR=status=active\n"
    )
    with open(PID_PATH, "w") as f:
        f.write(pid_info)

    print(f"[Launcher] HereBot PID: {_bot_proc.pid}", flush=True)
    print(f"[Launcher] PID lock: {PID_PATH}", flush=True)
    print(f"[Launcher] Log: {LOG_PATH}", flush=True)

    # Hold write-end of FIFO open to prevent bot from getting EOF
    fd = os.open(FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)

    try:
        while True:
            ret = _bot_proc.poll()
            if ret is not None:
                print(f"[Launcher] Bot exited code={ret}", flush=True)
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.close(fd)
        except:
            pass
        cleanup()

if __name__ == "__main__":
    main()
