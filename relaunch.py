"""Wait for the dying CryptoMind process to release the port, then start uvicorn.

Spawned by POST /api/control/restart when no supervisor is wrapping the
server (the usual case on Windows: `uvicorn` / `python -m uvicorn` launched
directly). Under run.sh / run.ps1 the supervisor relaunches instead.
"""
import os, socket, subprocess, sys, time, traceback

CREATE_NEW_CONSOLE = 0x00000010
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _log(msg):
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relaunch.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def _alive(pid):
    """True if pid is still running. os.kill(pid, 0) is NOT safe on Windows
    (it TerminateProcess's the pid)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return False
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_busy(host, port):
    probe = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    s = socket.socket()
    s.settimeout(0.4)
    try:
        s.connect((probe, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    parent = int(sys.argv[1])
    host = sys.argv[2]
    port = int(sys.argv[3])
    root = sys.argv[4]
    py = sys.argv[5]
    _log(f"waiting for pid {parent} to exit; then {py} uvicorn {host}:{port} in {root}")

    deadline = time.time() + 40
    while _alive(parent) and time.time() < deadline:
        time.sleep(0.2)
    if _alive(parent):
        _log(f"parent {parent} still alive after timeout — starting anyway")
    time.sleep(0.5)
    deadline = time.time() + 20
    while _port_busy(host, port) and time.time() < deadline:
        time.sleep(0.25)
    if _port_busy(host, port):
        _log(f"port {port} still busy — bind will likely fail")

    os.chdir(root)
    cmd = [py, "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]
    if os.name == "nt":
        flags = (CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP
                 | CREATE_BREAKAWAY_FROM_JOB)
        try:
            subprocess.Popen(cmd, cwd=root, creationflags=flags, close_fds=True)
        except OSError:
            flags = CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(cmd, cwd=root, creationflags=flags, close_fds=True)
        _log("spawned uvicorn in new console")
    else:
        os.execv(py, cmd)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log("relaunch failed:\n" + traceback.format_exc())
        raise
