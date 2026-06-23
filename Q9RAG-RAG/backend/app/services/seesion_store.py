import time
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

# 统一时区处理：UTC ISO8601
def get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class SessionStore(ABC):
    @abstractmethod
    def create_session(self) -> str: pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]: pass

    @abstractmethod
    def append_message(self, session_id: str, role: str, content: str): pass

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]: pass

    @abstractmethod
    def touch(self, session_id: str): pass
class InMemorySessionStore(SessionStore):
    def __init__(self, ttl: int = 3600):
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.storage[session_id] = {
            "messages": [],
            "created_at": get_utc_now(),
            "last_accessed": time.time()
        }
        return session_id

    def _is_expired(self, session_id: str) -> bool:
        if session_id not in self.storage: return True
        return (time.time() - self.storage[session_id]["last_accessed"]) > self.ttl

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._is_expired(session_id):
            self.storage.pop(session_id, None)
            return None
        self.touch(session_id)
        return self.storage[session_id]

    def append_message(self, session_id: str, role: str, content: str):
        if session_id in self.storage:
            message = {
                "role": role,
                "content": content,
                "timestamp": get_utc_now()
            }
            self.storage[session_id]["messages"].append(message)
            self.touch(session_id)

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        session = self.get_session(session_id)
        return session["messages"][-limit:] if session else []

    def touch(self, session_id: str):
        if session_id in self.storage:
            self.storage[session_id]["last_accessed"] = time.time()
import redis

class RedisSessionStore(SessionStore):
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.client = redis_client
        self.ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        data = {
            "messages": json.dumps([]),
            "created_at": get_utc_now()
        }
        self.client.hset(self._key(session_id), mapping=data)
        self.client.expire(self._key(session_id), self.ttl)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self.client.hgetall(self._key(session_id))
        if not data:
            return None
        self.touch(session_id)
        return {
            "messages": json.loads(data[b"messages"].decode("utf-8")),
            "created_at": data[b"created_at"].decode("utf-8")
        }

    def append_message(self, session_id: str, role: str, content: str):
        session = self.get_session(session_id)
        if session:
            new_msg = {"role": role, "content": content, "timestamp": get_utc_now()}
            session["messages"].append(new_msg)
            self.client.hset(self._key(session_id), "messages", json.dumps(session["messages"]))
            self.touch(session_id)

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        session = self.get_session(session_id)
        return session["messages"][-limit:] if session else []

    def touch(self, session_id: str):
        self.client.expire(self._key(session_id), self.ttl)