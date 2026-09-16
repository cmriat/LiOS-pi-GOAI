"""CPU socket regressions for the supervised server's port preflight."""

import errno
import socket
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def check_endpoint(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/inference/goai"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("goai_serve_port_test", scripts / "serve_xpolicylab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_endpoint_available


@pytest.fixture(params=[socket.AF_INET, socket.AF_INET6], ids=["ipv4", "ipv6"])
def endpoint(request):
    family = request.param
    host = "127.0.0.1" if family == socket.AF_INET else "::1"
    if family == socket.AF_INET6:
        try:
            with socket.socket(family) as probe:
                probe.bind((host, 0))
        except OSError:
            pytest.skip("IPv6 loopback is unavailable")
    return family, host


def test_check_does_not_keep_port_bound(check_endpoint, endpoint):
    family, host = endpoint
    with socket.socket(family) as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]
    check_endpoint(host, port)
    with socket.socket(family) as listener:
        listener.bind((host, port))
        listener.listen()


def test_live_listener_is_still_rejected(check_endpoint, endpoint):
    family, host = endpoint
    with socket.socket(family) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        listener.listen()
        with pytest.raises(OSError) as caught:
            check_endpoint(host, listener.getsockname()[1])
        assert caught.value.errno == errno.EADDRINUSE


def test_immediate_restart_with_time_wait(check_endpoint, endpoint):
    family, host = endpoint
    with socket.socket(family) as listener, socket.socket(family) as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, 0))
        port = listener.getsockname()[1]
        listener.listen()
        listener.settimeout(2)
        client.settimeout(2)
        client.connect((host, port))
        accepted, _ = listener.accept()
        with accepted:
            accepted.settimeout(2)
            # The server closes first so TIME_WAIT belongs to its listening port.
            accepted.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
            client.shutdown(socket.SHUT_WR)
            assert accepted.recv(1) == b""
    # Establish that the old non-reusing preflight would reject this port.
    with socket.socket(family) as old_preflight:
        with pytest.raises(OSError) as caught:
            old_preflight.bind((host, port))
        assert caught.value.errno == errno.EADDRINUSE
    check_endpoint(host, port)
    with socket.socket(family) as restarted:
        restarted.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        restarted.bind((host, port))
        restarted.listen()
