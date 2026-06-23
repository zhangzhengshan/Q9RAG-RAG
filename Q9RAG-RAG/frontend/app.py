import streamlit as st
import requests
import time
import re

# ---------- 基础配置 ----------
BACKEND_URL = "http://127.0.0.1:8080"
st.set_page_config(page_title="LLM 多轮对话系统", page_icon="💬", layout="wide")


# ---------- 工具函数：清洗 Markdown 标签（保证渲染正常） ----------
def clean_markdown(text):
    """清洗未闭合的 Markdown 标签，避免渲染错乱"""
    if not text:
        return ""
    # 移除未闭合的粗体/斜体标签
    text = re.sub(r'\*\*(?!\*+\*\*)', '', text)
    text = re.sub(r'\*(?!\*+\*)', '', text)
    # 移除未闭合的链接/图片标签
    text = re.sub(r'\[(?![^\]]*\])', '', text)
    text = re.sub(r'\((?![^\)]*\))', '', text)
    # 移除多余的换行和空格
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------- Session 初始化 ----------
# 会话ID - 用于多轮对话上下文管理
if "session_id" not in st.session_state:
    r = requests.post(f"{BACKEND_URL}/api/sessions", timeout=10)
    st.session_state.session_id = r.json().get("session_id") if r.ok else None

# 对话历史 - 存储所有消息记录
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- 页面布局 ----------
st.title("💬 LLM 多轮对话系统")
st.caption("专注流畅的多轮对话体验，完整展示所有回答内容")

# 清空对话按钮
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.divider()

# ---------- 展示多轮对话历史 ----------
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        # 展示用户消息
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        # 展示助手消息（完整展示，带滚动）
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                # 完整展示回答内容，超出高度可滚动
                cleaned_content = clean_markdown(message["content"])
                st.markdown(
                    f"""
                    <div style="max-height: 700px; overflow-y: auto; padding: 12px; border-radius: 10px; background-color: #f0f2f6;">
                        {cleaned_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 展示参考资料（如果有）
                if message.get("sources") and len(message["sources"]) > 0:
                    with st.expander("📚 参考资料", expanded=False):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"{idx}. {source}")

# ---------- 聊天输入框 ----------
st.divider()
user_input = st.chat_input("请输入你的问题，按回车发送...", key="chat_input")

# ---------- 处理用户输入 ----------
if user_input:
    # 将用户消息添加到对话历史
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "sources": []
    })

    # 实时展示用户消息
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

    # 发送请求获取回答
    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 正在思考中..."):
                # 发送请求到后端
                start_time = time.time()
                resp = requests.post(
                    f"{BACKEND_URL}/api/sessions/{st.session_state.session_id}/messages",
                    json={"query": user_input},
                    timeout=60,
                    stream=False
                )
                resp.encoding = "utf-8"

                # 解析响应数据
                response_data = resp.json()
                answer = clean_markdown(str(response_data.get("answer", "")).strip())
                sources = response_data.get("sources", [])

                # 补足短暂等待时间，提升用户体验
                elapsed_time = time.time() - start_time
                if elapsed_time < 1:
                    time.sleep(1 - elapsed_time)

            # 完整展示助手回答
            st.markdown(
                f"""
                <div style="max-height: 700px; overflow-y: auto; padding: 12px; border-radius: 10px; background-color: #f0f2f6;">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            # 展示参考资料
            if sources and len(sources) > 0:
                with st.expander("📚 参考资料", expanded=False):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"{idx}. {source}")

        # 将助手消息添加到对话历史
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    # 自动滚动到最新消息
    st.rerun()