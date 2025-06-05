import threading
import http.server
import socketserver
import time
from core.tools.tool_registry import file_read, internet_fetch, os_metrics, repo_scan


def test_file_read(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello")
    res = file_read(str(f))
    assert res["status"] == "success"
    assert res["output"] == "hello"


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
    http.server.HTTPServer


def test_os_metrics():
    res = os_metrics()
    assert isinstance(res, dict)
    assert res or res.get("error")
