import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import httpx

from backend.main import app
from backend.app.services.llm_client import DeepseekServiceError


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client



def test_deepseek_500_returns_502_and_systemerror(client, monkeypatch, caplog):
    """
    Test D - 错误场景：Deepseek 返回 500
    期望：
    1. HTTP 返回 502
    2. session history 中插入 systemerror
    3. logs 中包含 request_id + timestamp
    """

    # ---------- 1. 创建 session ----------
    resp = client.post("/api/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # ---------- 2. mock LLMClient.call_llm 抛出 DeepseekServiceError ----------
    fake_request_id = "test-request-id-500"
    fake_timestamp = datetime.now(timezone.utc).isoformat()

    async def mock_call_llm(*args, **kwargs):
        raise DeepseekServiceError(
            request_id=fake_request_id,
            status_code=500,
            message="mock deepseek internal error",
            timestamp=fake_timestamp,
        )

    # monkeypatch 替换实例方法
    monkeypatch.setattr(
        "backend.app.services.llm_client.LLMClient.call_llm",
        mock_call_llm
    )

    # ---------- 3. 发送对话请求 ----------
    with caplog.at_level("ERROR"):
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"query": "测试 Deepseek 500 场景"}
        )

    # ---------- 4. 断言 HTTP 502 ----------
    assert resp.status_code == 502
    assert "Bad Gateway" in resp.text

    # ---------- 5. 校验 session history ----------
    history_resp = client.get(
        f"/api/sessions/{session_id}/history"
    )
    assert history_resp.status_code == 200

    history = history_resp.json()["history"]
    assert len(history) >= 2

    # user 消息仍然存在（关键：session 不受影响）
    assert history[0]["role"] == "user"
    assert "测试 Deepseek 500 场景" in history[0]["content"]

    # systemerror 被写入
    system_msgs = [m for m in history if m["role"] == "system"]
    assert len(system_msgs) == 1

    system_msg = system_msgs[0]["content"]
    assert "[systemerror]" in system_msg
    assert fake_request_id in system_msg
    assert fake_timestamp in system_msg

    # ---------- 6. 校验日志 ----------
    error_logs = "\n".join(record.message for record in caplog.records)

    assert fake_request_id in error_logs
    assert "DeepseekServiceError" in error_logs or "Deepseek" in error_logs
