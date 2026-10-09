import os
from pathlib import Path

os.environ["AWS_ENABLED"] = "0"
os.environ["DATABASE_PATH"] = str(Path(__file__).parent / "test-assistant.db")

from fastapi.testclient import TestClient

import main


def setup_module():
    Path(os.environ["DATABASE_PATH"]).unlink(missing_ok=True)
    main.init_database()


def teardown_module():
    Path(os.environ["DATABASE_PATH"]).unlink(missing_ok=True)


def test_health_and_static_frontend():
    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["region"] == "us-east-1"
        assert health.json()["services"]["bedrock"] == "disabled"
        page = client.get("/")
        assert page.status_code == 200
        assert "Click to start" in page.text


def test_persistent_task_lifecycle_and_summary():
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "connected"
            socket.send_json({"type": "hello", "client_id": "test-user-123", "timezone": "UTC"})
            ready = socket.receive_json()
            assert ready["type"] == "ready"

            socket.send_json({"type": "chat", "text": "remind me to review the plan tomorrow at 9 am"})
            created = socket.receive_json()
            assert created["type"] == "reply"
            assert created["speech"]["provider"] == "browser"
            assert created["tasks"][0]["title"] == "review the plan"
            task_id = created["tasks"][0]["id"]

            socket.send_json({"type": "task_toggle", "id": task_id, "completed": True})
            completed = socket.receive_json()
            assert completed["tasks"][0]["completed"] is True

            socket.send_json({"type": "summary"})
            summary = socket.receive_json()
            assert "completed 1 task today" in summary["text"]

            socket.send_json({"type": "task_delete", "id": task_id})
            confirmation = socket.receive_json()
            assert confirmation["type"] == "confirm"
            assert confirmation["tasks"]

            socket.send_json({"type": "confirm", "token": confirmation["token"]})
            deleted = socket.receive_json()
            assert deleted["tasks"] == []


def test_message_limits_and_local_fallback():
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "hello", "client_id": "another-user", "timezone": "UTC"})
            socket.receive_json()
            socket.send_json({"type": "chat", "text": "What can you do?"})
            reply = socket.receive_json()
            assert reply["type"] == "reply"
            assert "manage persistent tasks" in reply["text"]
            assert len(reply["text"]) <= 500
