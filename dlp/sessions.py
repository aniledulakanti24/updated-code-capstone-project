import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class SessionState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionData:
    session_id: str
    state: SessionState
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any]
    memory: List[Dict[str, Any]]
    metrics: Dict[str, int]


class InMemorySessionService:
    """Session management for agent state persistence"""
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.logger = logging.getLogger("SessionService")

    def create_session(self, session_id: str) -> SessionData:
        session = SessionData(
            session_id=session_id,
            state=SessionState.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            context={},
            memory=[],
            metrics={"scans": 0, "detections": 0, "actions": 0}
        )
        self.sessions[session_id] = session
        self.logger.info(f"Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, updates: Dict[str, Any]):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.updated_at = datetime.now()
            session.context.update(updates.get("context", {}))
            if "memory" in updates:
                session.memory.extend(updates["memory"])
            if "metrics" in updates:
                for key, value in updates["metrics"].items():
                    session.metrics[key] = session.metrics.get(key, 0) + value

    def pause_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state = SessionState.PAUSED
            self.logger.info(f"Paused session: {session_id}")

    def resume_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state = SessionState.ACTIVE
            self.logger.info(f"Resumed session: {session_id}")


class LongRunningOperation:
    """Support for pausable/resumable operations"""

    def __init__(self, operation_id: str, session_service: InMemorySessionService):
        self.operation_id = operation_id
        self.session_service = session_service
        self.checkpoints: List[Dict] = []
        self.logger = logging.getLogger("LongRunningOp")

    def create_checkpoint(self, state: Dict):
        """Save operation state"""
        checkpoint = {
            "checkpoint_id": len(self.checkpoints),
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Checkpoint created: {checkpoint['checkpoint_id']}")

    def pause(self, session_id: str):
        """Pause operation"""
        self.session_service.pause_session(session_id)
        self.logger.info(f"Operation paused: {self.operation_id}")

    def resume(self, session_id: str) -> Dict:
        """Resume from last checkpoint"""
        self.session_service.resume_session(session_id)
        if self.checkpoints:
            last_checkpoint = self.checkpoints[-1]
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint['checkpoint_id']}")
            return last_checkpoint["state"]
        return {}
