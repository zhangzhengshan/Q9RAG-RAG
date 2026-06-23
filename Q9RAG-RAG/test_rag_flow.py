import requests
import time

# 配置FastAPI完整基础URL（无需认证）
BASE_URL = "http://127.0.0.1:8000"

def test_rag_chat_flow():
    print("开始 RAG 功能验收测试...")
    session_id = None

    # 1. 创建会话 (POST /api/sessions)
    try:
        session_res = requests.post(f"{BASE_URL}/api/sessions")
        session_res.raise_for_status()  # 主动抛出HTTP错误
        session_data = session_res.json()
        session_id = session_data.get("session_id")

        if not session_id:
            print("❌ 步骤1失败: 创建会话成功，但未返回session_id")
            return
        print(f"✅ 步骤1成功: 获取到 Session ID: {session_id}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 步骤1失败: 无法创建会话，错误信息: {str(e)}")
        return

    # 2. 发送提问 (POST /api/sessions/{session_id}/messages)
    # 仅使用role和query两个参数，符合要求
    payload = {
        "role": "user",
        "query": "为什么老人不喜欢去医院？"
    }
    try:
        msg_res = requests.post(
            f"{BASE_URL}/api/sessions/{session_id}/messages",
            json=payload  # 自动设置Content-Type: application/json
        )
        msg_res.raise_for_status()
        msg_data = msg_res.json()

        # 提取接口返回的字段（对应FastAPI接口的answer和sources）
        answer_text = msg_data.get("answer", "")
        retrieved_docs = msg_data.get("sources", [])

        # 验证回答长度
        if len(answer_text) > 10:
            print(f"✅ 步骤2成功: 回答内容长度符合要求 ({len(answer_text)} 字)")
            print(f"💡 回答内容: {answer_text}")
        else:
            print(f"❌ 步骤2验证失败: 回答内容过短（长度：{len(answer_text)}）")
            print(f"💡 回答内容: {answer_text}")

        # 打印检索到的文档数量
        retrieved_count = len(retrieved_docs)
        print(f"💡 检索到的文档数量: {retrieved_count}")
        if retrieved_count > 0:
            print(f"💡 检索文档预览: {[doc[:50] + '...' if isinstance(doc, str) else doc for doc in retrieved_docs]}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 步骤2失败: 发送消息出错，错误信息: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"💡 服务器返回状态码: {e.response.status_code}")
            print(f"💡 服务器返回详情: {e.response.text}")
        return

    # 关键：添加延时，规避异步写入后立即查询的延迟问题
    print("\n⏳ 等待对话记录持久化...")
    time.sleep(2)  # 延时2秒，可根据实际情况调整（1-5秒均可）

    # 3. 验证历史记录 (GET /api/sessions/{session_id}/history)
    # 对应接口返回格式：{"history": history}
    try:
        history_res = requests.get(
            f"{BASE_URL}/api/sessions/{session_id}/history",
            # 可指定limit参数（默认10，此处显式传入保持清晰）
            params={"limit": 10}
        )
        history_res.raise_for_status()
        history_data = history_res.json()
        # 提取历史记录列表（匹配接口返回格式）
        chat_history = history_data.get("history", [])
        history_count = len(chat_history)

        print(f"\n✅ 步骤3成功: 获取历史记录接口调用正常")
        print(f"💡 历史记录总条数: {history_count}")

        # 验证历史记录条数（用户+助手消息，至少2条）
        if history_count >= 2:
            print(f"✅ 步骤3验证成功: 历史记录条数符合要求（≥2条）")
            # 验证对话角色顺序（先user，后assistant）
            first_msg = chat_history[0]
            second_msg = chat_history[1]
            if first_msg.get("role") == "user" and second_msg.get("role") == "assistant":
                print("✅ 历史记录角色顺序验证通过")
            else:
                print(f"⚠️  历史记录角色顺序异常: 第一条角色={first_msg.get('role')}, 第二条角色={second_msg.get('role')}")
        else:
            print(f"❌ 步骤3验证失败: 历史记录条数不足（当前{history_count}条，要求≥2条）")

        # 打印完整历史记录，便于排查
        if chat_history:
            print(f"\n💡 完整历史记录:")
            for idx, msg in enumerate(chat_history):
                role = msg.get("role", "未知角色")
                content = msg.get("content", "") or msg.get("query", "") or msg.get("answer", "")
                print(f"  第{idx+1}条 - 角色: {role}, 内容: {content[:100]}...")

    except requests.exceptions.RequestException as e:
        print(f"❌ 步骤3失败: 无法获取历史记录，错误信息: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"💡 服务器返回状态码: {e.response.status_code}")
            print(f"💡 服务器返回详情: {e.response.text}")
        return

if __name__ == "__main__":
    test_rag_chat_flow()