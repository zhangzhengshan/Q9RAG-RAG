import os
import json
import asyncio
import uuid

import httpx
from typing import List, Dict, Any
import logging
from datetime import datetime, timezone

# 配置简单的日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Master_LLM")
class DeepseekServiceError(Exception):
    """
    当 Deepseek 返回 500 时抛出，携带 request_id, status_code, timestamp, message
    """
    def __init__(self, request_id: str, status_code: int, message: str, timestamp: str):
        super().__init__(message)
        self.request_id = request_id
        self.status_code = status_code
        self.message = message
        self.timestamp = timestamp

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model_name = "deepseek-chat"

        if not self.api_key:
            raise ValueError("❌ [RAG大师] 错误：未检测到 DEEPSEEK_API_KEY 环境变量")

    @staticmethod
    def build_chat_messages(
            query: str,
            retrieved_docs: List[str],
            chat_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        构建支持多轮对话的 messages，遵循神哲学家角色与检索提示逻辑
        """
        system_prompt = (
            "你是一个久历人世，心龄三千年，见过生老病死，盛衰兴亡的神哲学家，"
            "你需要使用你以前写下的文章，来回忆你是谁，并且用这些文章里的知识来指出我的问题所在和解决办法。"
            "请尽可能有条理地，优美的地回答。必要时请引用原文。如果我的情绪明显不佳，请你更温柔的说话，谢谢。"
            "\n\n【重要规则】如果收到系统提示“当前未检索到任何相关文档”，请直接回复“无检索结果”，不要添加任何额外内容。"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 注入历史对话（短期记忆）
        for msg in chat_history:
            messages.append(msg)

        # 情况 1：未检索到任何内容
        if not retrieved_docs:
            messages.append({
                "role": "user",
                "content": f"{query}\n\n（系统提示：当前未检索到任何相关文档）"
            })
            return messages

        # 情况 2：有检索结果
        context_text = "\n\n".join(
            f"[文档 {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)
        )

        messages.append({
            "role": "user",
            "content": (
                "以下是从知识库中检索到的相关内容：\n\n"
                f"{context_text}\n\n"
                f"用户问题：{query}"
            )
        })

        return messages

    async def call_llm(self, messages: List[Dict[str, str]], temperature=0.3, max_tokens=2048) -> str:
        """
        异步调用 DeepSeek API，包含错误处理与 3 次重试逻辑
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        max_retries = 3
        backoff_factor = 2  # 重试间隔倍数

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        json=payload
                    )
                    # **特判 500：立即记录并抛出专用异常（不按普通 HTTPStatusError 重试逻辑）**
                    if response.status_code == 500:
                        # 尝试从 header 里取 request id（若没有则生成一个），并记录 timestamp
                        request_id = response.headers.get("X-Request-Id") or response.headers.get(
                            "X-Request-ID") or str(uuid.uuid4())
                        timestamp = datetime.now(timezone.utc).isoformat()
                        body_text = response.text
                        logger.error(
                            f"❌ [RAG大师][Deepseek 500] request_id={request_id} timestamp={timestamp} "
                            f"status=500 body={body_text}"
                        )
                        # 抛出自定义异常，上层（main）负责把 systemerror 写入 session 并返回 502
                        raise DeepseekServiceError(request_id=request_id, status_code=500, message=body_text,
                                                   timestamp=timestamp)

                    # 其它非 2xx 会触发 raise_for_status -> 捕获到 HTTPStatusError 走重试逻辑
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

                except DeepseekServiceError:
                    # 直接向上抛出，不做重试（已经记录日志）
                    raise

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else None
                    # 对于 500 我们已在上面处理，这里处理其他 status（可重试）
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"⚠️ [RAG大师] HTTPStatusError (尝试 {attempt + 1}/{max_retries}) status={status} error={e}. {wait_time}s 后重试...")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(wait_time)

                except httpx.RequestError as e:
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"⚠️ [RAG大师] RequestError (尝试 {attempt + 1}/{max_retries}): {e}. {wait_time}s 后重试...")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(wait_time)
        return "抱歉，由于冥想空间连通不畅（API错误），我暂时无法回答。"