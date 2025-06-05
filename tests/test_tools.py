import threading
import http.server
import socketserver
import time
import sys, types
sys.modules.setdefault('faiss', types.SimpleNamespace())
sys.modules.setdefault('ollama', types.SimpleNamespace())
torch_stub = types.ModuleType('torch')
torch_stub.nn = types.ModuleType('nn')
torch_stub.nn.functional = types.ModuleType('functional')
sys.modules.setdefault('torch', torch_stub)
sys.modules.setdefault('torch.nn', torch_stub.nn)
sys.modules.setdefault('torch.nn.functional', torch_stub.nn.functional)
sys.modules.setdefault('yaml', types.SimpleNamespace())
import http.server
import socketserver
from core.tools.tool_registry import (
    file_read,
    read_file,
    internet_fetch,
    fetch_url,
    os_metrics,
    get_system_metrics,
    repo_scan,
    run_shell,
)
def test_file_read(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello")
    res = file_read(str(f))
    assert res["status"] == "success"
    assert res["output"] == "hello"

    alias = read_file(str(f))
    assert alias["output"] == "hello"


def test_repo_scan(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a")
    monkeypatch.chdir(tmp_path)
    res = repo_scan("*.txt")
    assert res["status"] == "success"
    assert "a.txt" in res["output"]


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_internet_fetch(tmp_path):
    # start simple http server serving a file
    f = tmp_path / "index.html"
    f.write_text("Hello World")
    def serve():
        handler = lambda *a, **kw: Handler(*a, directory=str(tmp_path), **kw)
        with socketserver.TCPServer(("", 0), handler) as httpd:
            port = httpd.server_address[1]
            thread.port = port
            httpd.serve_forever()
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    while not hasattr(thread, "port"):
        time.sleep(0.1)
    url = f"http://localhost:{thread.port}/index.html"
    res = internet_fetch(url)
    assert res["status"] == "success"
    assert "Hello World" in res["output"]
    alias = fetch_url(url)
    assert alias["status"] == "success"
    http.server.HTTPServer


def test_os_metrics():
    res = os_metrics()
    assert isinstance(res, dict)
    assert res or res.get("error")


def test_run_shell():
    res = run_shell("echo hi")
    assert "output" in res


def test_get_system_metrics():
    res = get_system_metrics()
    assert isinstance(res, dict)


