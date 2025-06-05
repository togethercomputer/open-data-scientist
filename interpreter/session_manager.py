import asyncio
from datetime import datetime
from typing import Dict, Any
import threading


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.namespace: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.lock = asyncio.Lock()


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create_session(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id)
            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())
