"""Thread-safe dialogue state and adaptive follow-up timing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class DialogueState(str, Enum):
    STANDBY = "STANDBY"
    AWAITING_COMMAND = "AWAITING_COMMAND"
    USER_SPEAKING = "USER_SPEAKING"
    MODEL_THINKING = "MODEL_THINKING"
    TOOL_RUNNING = "TOOL_RUNNING"
    ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"
    INTERRUPTED = "INTERRUPTED"
    MUTED = "MUTED"
    ERROR = "ERROR"


_UI_MAP = {
    DialogueState.STANDBY: "STANDBY",
    DialogueState.AWAITING_COMMAND: "LISTENING",
    DialogueState.USER_SPEAKING: "LISTENING",
    DialogueState.MODEL_THINKING: "THINKING",
    DialogueState.TOOL_RUNNING: "THINKING",
    DialogueState.ASSISTANT_SPEAKING: "SPEAKING",
    DialogueState.FOLLOW_UP: "FOLLOW_UP",
    DialogueState.INTERRUPTED: "LISTENING",
    DialogueState.MUTED: "MUTED",
    DialogueState.ERROR: "SLEEPING",
}


@dataclass(frozen=True)
class StateSnapshot:
    state: DialogueState
    changed_at: float
    reason: str


class ConversationStateManager:
    def __init__(self, on_change: Callable[[str], None] | None = None):
        self._lock = threading.RLock()
        self._snapshot = StateSnapshot(DialogueState.STANDBY, time.monotonic(), "startup")
        self._on_change = on_change

    @property
    def state(self) -> DialogueState:
        with self._lock:
            return self._snapshot.state

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return self._snapshot

    def set(self, state: DialogueState, reason: str = "") -> None:
        with self._lock:
            if state == self._snapshot.state and not reason:
                return
            self._snapshot = StateSnapshot(state, time.monotonic(), reason)
        if self._on_change:
            self._on_change(_UI_MAP[state])

    @staticmethod
    def follow_up_seconds(
        transcript: str,
        base: float = 10.0,
        tool_used: bool = False,
        vision_used: bool = False,
    ) -> float:
        """Choose a natural follow-up window from the completed turn."""
        text = (transcript or "").strip().lower()
        if vision_used:
            return max(base, 15.0)
        if tool_used:
            return max(base, 12.0)
        if not text:
            return base
        if "?" in text or any(
            token in text
            for token in ("why", "how", "what", "which", "explain", "compare", "think")
        ):
            return max(base, 12.0)
        words = len(text.split())
        if words <= 6:
            return min(base, 6.0)
        return base
