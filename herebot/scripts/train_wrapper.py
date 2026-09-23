#!/usr/bin/env python3
"""
HereBot Training Wrapper - runs in-process, polls /tmp/herebot_cmd for input.
Write a line to /tmp/herebot_cmd -> bot processes it -> output in /tmp/herebot_out
"""
import os, sys, time, signal, traceback

REPO = "/home/hrirake/herebot"
PID_PATH = os.path.join(REPO, "herebot/curiosity_state/training.pid")
CMD_FILE = "/tmp/herebot_cmd"
OUT_FILE = "/tmp/herebot_out"
POLL_INTERVAL = 0.5

sys.path.insert(0, os.path.join(REPO, "herebot"))

_running = True

def _stop(signum, frame):
    global _running
    _running = False

def main():
    global _running
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # Clean stale files
    for f in (CMD_FILE, OUT_FILE):
        if os.path.exists(f):
            os.remove(f)

    # Import and start engine
    from main import CuriosityEngine, IdleReasoner

    print("[Wrapper] Starting HereBot engine...", flush=True)
    engine = CuriosityEngine(load_state=True)
    idle = IdleReasoner(engine, short_idle_threshold=8.0, long_idle_threshold=600.0)
    engine._idle_reasoner = idle
    idle.start()
    print("[Wrapper] Engine started. Entering poll loop.", flush=True)

    # Write PID lock
    pid_info = (
        f"PID={os.getpid()}\n"
        f"START_TIME={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"SESSION_ID=training_{int(time.time())}\n"
        f"REPO={REPO}\n"
        f"REPAIR=status=active\n"
        f"MODE=in-process\n"
    )
    with open(PID_PATH, "w") as f:
        f.write(pid_info)
    print(f"[Wrapper] PID lock: {PID_PATH}", flush=True)

    last_mtime = 0
    while _running:
        try:
            if os.path.exists(CMD_FILE):
                mtime = os.path.getmtime(CMD_FILE)
                if mtime > last_mtime:
                    last_mtime = mtime
                    with open(CMD_FILE, "r") as f:
                        line = f.read().strip()
                    if not line:
                        time.sleep(POLL_INTERVAL)
                        continue
                    if line.lower() in ("exit", "quit"):
                        with open(OUT_FILE, "w") as f:
                            f.write("Shutting down.\n")
                        break

                    result = engine.process_input(line)
                    output_lines = []
                    output_lines.append("--- Curiosity Output ---")
                    for l in result.get("curiosity_artifacts", []):
                        output_lines.append(l)
                    output_lines.append("------------------------")
                    response = "\n".join(output_lines)
                    with open(OUT_FILE, "w") as f:
                        f.write(response + "\n")
                    print(f"[Wrapper] Processed: {line[:60]!r}", flush=True)

            time.sleep(POLL_INTERVAL)
        except Exception:
            traceback.print_exc()
            time.sleep(2)

    # Cleanup
    idle.running = False
    try:
        engine._save_state()
    except:
        pass
    try:
        engine.tracker.save_data()
    except:
        pass
    for f in (CMD_FILE, OUT_FILE, PID_PATH):
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass
    print("[Wrapper] Shutdown complete.", flush=True)

if __name__ == "__main__":
    main()
