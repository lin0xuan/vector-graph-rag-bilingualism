import socket
import os
import pytest


@pytest.fixture(autouse=True)
def no_remote_network(monkeypatch):
    """Tests can use local Milvus/HTTP but cannot call paid endpoints."""
    original = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError("Remote network forbidden in tests")
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def milvus_server_uri():
    uri = os.getenv("VGRAG_TEST_MILVUS_URI")
    if not uri:
        pytest.skip("Set VGRAG_TEST_MILVUS_URI to run real Standalone integration tests")
    from urllib.parse import urlsplit
    if urlsplit(uri).hostname not in ("127.0.0.1", "localhost", "::1"):
        pytest.fail("Integration tests require a local Standalone server")
    return uri
