"""Cancellable tool jobs and confirmation policy."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class RiskLevel(str, Enum):
    SAFE = "safe"
    DISRUPTIVE = "disruptive"
    SENSITIVE = "sensitive"


@dataclass
class ToolJob:
    id: str
    name: str
    args: dict[str, Any]
    risk: RiskLevel
    created_at: float = field(default_factory=time.monotonic)
    status: str = "pending"
    result: Any = None
    error: str | None = None
    task: asyncio.Task | None = None


class ToolJobManager:
    """Tracks tool work and supports model-issued cancellation."""

    def __init__(self):
        self._jobs: dict[str, ToolJob] = {}
        self._by_call_id: dict[str, str] = {}

    @staticmethod
    def classify(name: str, args: dict[str, Any]) -> RiskLevel:
        if name in {"send_message", "shutdown_jarvis"}:
            return RiskLevel.SENSITIVE
        if name == "file_controller":
            action = str(args.get("action", "")).lower()
            if any(word in action for word in ("delete", "remove", "trash", "overwrite")):
                return RiskLevel.SENSITIVE
            if any(word in action for word in ("move", "rename", "compress", "extract")):
                return RiskLevel.DISRUPTIVE
        action = str(args.get("action", "")).lower().strip()
        action_text = " ".join(str(v) for v in args.values()).lower()
        if name == "computer_settings":
            if any(word in action_text for word in ("shutdown", "restart", "log out", "lock screen", "close all")):
                return RiskLevel.SENSITIVE
            if any(word in action_text for word in ("close app", "close window", "wifi off", "disable wifi")):
                return RiskLevel.DISRUPTIVE
            return RiskLevel.SAFE
        if name == "desktop_control":
            if action == "clean" or "delete" in action_text:
                return RiskLevel.SENSITIVE
            if action in {"organize", "wallpaper", "wallpaper_url", "task"}:
                return RiskLevel.DISRUPTIVE
            return RiskLevel.SAFE
        if name == "browser_control" and action in {"close", "close_all"}:
            return RiskLevel.DISRUPTIVE
        # Direct typing/clicking is intentionally immediate; confirmations are
        # reserved for actions with broad or hard-to-reverse effects.
        return RiskLevel.SAFE

    @staticmethod
    def confirmation_prompt(name: str, args: dict[str, Any], risk: RiskLevel) -> str:
        details = ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, ""))
        if len(details) > 180:
            details = details[:177] + "..."
        return (
            f"Confirmation required before {risk.value} action '{name}'"
            + (f" ({details})" if details else "")
            + ". Ask the user a brief, explicit yes/no question and do not call the tool again until they confirm."
        )

    def create(self, call_id: str, name: str, args: dict[str, Any]) -> ToolJob:
        job = ToolJob(
            id=uuid.uuid4().hex[:12],
            name=name,
            args=dict(args),
            risk=self.classify(name, args),
        )
        self._jobs[job.id] = job
        self._by_call_id[str(call_id)] = job.id
        return job

    async def run(self, job: ToolJob, operation: Callable[[], Awaitable[Any]]) -> Any:
        job.status = "running"
        job.task = asyncio.create_task(operation(), name=f"tool:{job.name}:{job.id}")
        try:
            job.result = await job.task
            job.status = "completed"
            return job.result
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            raise

    def cancel_call(self, call_id: str) -> bool:
        job_id = self._by_call_id.get(str(call_id))
        job = self._jobs.get(job_id or "")
        if not job or not job.task or job.task.done():
            return False
        job.task.cancel()
        job.status = "cancelling"
        return True

    def cancel_all(self) -> int:
        count = 0
        for job in self._jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()
                job.status = "cancelling"
                count += 1
        return count
