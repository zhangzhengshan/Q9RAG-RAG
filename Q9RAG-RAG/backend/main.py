import os
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
# 导入你之前定义的组件
from backend.app.services.seesion_store import InMemorySessionStore, RedisSessionStore
from backend.app.services.retrieval import Retriever
from backend.app.services.llm_client import LLMClient
from backend.app.services.llm_client import LLMClient, DeepseekServiceError, logger as llm_logger
from datetime import datetime, timezone

# --- 配置与常量 ---
CONFIG = {
    "metadata_path": r"F:\pyproject\Q9RAG\backend\scripts\metadatatwo.json",
    "index_path": r"F:\pyproject\Q9RAG\backend\scripts\vector_index.bin",
    "embedding_model_path": "BAAI/bge-small-zh-v1.5",
    "reranker_model_path": "BAAI/bge-reranker-base",
    "torch_threads": 6,
    "retrieval_mode": "hybrid",
    "sim_threshold": 0.45,
    "top_k_initial": 8,
    "top_k_bm25": 8,
    "hybrid_pool_size": 12,
    "vector_weight": 0.7,
    "bm25_weight": 0.3,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "rrf_k": 60,
    "faiss_ef_search": 64,
    "final_n": 3,
    "score_filter": -0.7
}
SESSION_TTL = 3600  # 60分钟


# --- 全局单例持有者 ---
class AppState:
    retriever: Retriever = None
    llm_client: LLMClient = None
    session_store: InMemorySessionStore = None


state = AppState()


# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：加载模型与索引
    print("正在初始化全栈 RAG 引擎...")
    state.retriever = Retriever.from_config(CONFIG)
    state.llm_client = LLMClient()
    # 默认使用内存存储，如需分布式可替换为 RedisSessionStore
    state.session_store = InMemorySessionStore(ttl=SESSION_TTL)
    yield
    # 关闭时：清理资源（如有必要）
    print(" 引擎已安全关闭。")


app = FastAPI(title="神哲学家 RAG 系统", lifespan=lifespan)


# --- 请求模型 ---
class MessageRequest(BaseModel):
    query: str


# --- 路由实现 ---

@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    session_id = state.session_store.create_session()
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str, limit: int = 10):
    """获取会话历史"""
    history = state.session_store.get_history(session_id, limit=limit)
    if not history and not state.session_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": history}


@app.post("/api/sessions/{session_id}/messages")
async def chat(session_id: str, req: MessageRequest):
    """核心 RAG 对话接口"""
    # 1. 验证并更新 Session 活跃时间 (TTL 延长)
    session = state.session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话已过期或不存在")

    # 先把用户消息持久化（保证即使上游失败，session 里仍有用户提问记录）
    state.session_store.append_message(session_id, "user", req.query)

    # 2. 检索增强 (Retrieval)
    history = state.session_store.get_history(session_id, limit=3)
    retrieved_docs = state.retriever.retrieve(req.query)

    # 3. 构建 Prompt (LLM Client)
    messages = state.llm_client.build_chat_messages(
        query=req.query,
        retrieved_docs=retrieved_docs,
        chat_history=history
    )

    # 4. 调用 LLM (Generation) —— 捕获 DeepseekServiceError 并以可控方式反馈
    try:
        answer = await state.llm_client.call_llm(messages)
    except DeepseekServiceError as e:
        # 在 session 历史中插入 systemerror message（便于前端 / 测试识别）
        system_msg = (
            f"[systemerror] upstream Deepseek 服务异常，status={e.status_code}，"
            f"request_id={e.request_id}，timestamp={e.timestamp}。"
            "（错误已记录）"
        )
        state.session_store.append_message(session_id, "system", system_msg)

        # 额外日志（主服务层）
        llm_logger.error(f"[Captured @ main] DeepseekServiceError request_id={e.request_id} timestamp={e.timestamp} message={e.message}")

        # 将上游 500 映射为 502 返回给客户端（Bad Gateway）
        raise HTTPException(status_code=502, detail="Bad Gateway: upstream LLM error")
    except Exception as e:
        # 兜底：记录并反馈 502，同时在 session 写入 systemerror
        timestamp = datetime.now(timezone.utc).isoformat()
        sys_msg = f"[systemerror] LLM 调用失败（非 500），timestamp={timestamp}，error={str(e)}"
        state.session_store.append_message(session_id, "system", sys_msg)
        llm_logger.exception("Unexpected error when calling LLM")
        raise HTTPException(status_code=502, detail="Bad Gateway: LLM call failed")

    # 5. 持久化 assistant 回答（正常路径）
    state.session_store.append_message(session_id, "assistant", answer)

    return {
        "answer": answer,
        "sources": retrieved_docs if retrieved_docs else []
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "engine": "RAG-Master-v1",
        "resources_loaded": state.retriever is not None
    }
