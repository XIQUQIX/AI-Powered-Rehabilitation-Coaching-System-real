
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.text_to_voice.tts import speak


@dataclass
class SpeechEvent:
    text: str
    tier_used: str


class SpeechManager:
    def __init__(self, enabled: bool = False, min_gap_seconds: float = 5.0):
        self.enabled = enabled
        self.min_gap_seconds = min_gap_seconds
        self._queue: "queue.Queue[SpeechEvent]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._last_spoken_at = 0.0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def enqueue(self, text: str, tier_used: str) -> bool:
        if not self.enabled or tier_used not in {"tier_2", "tier_3"}:
            return False
        try:
            self._queue.put_nowait(SpeechEvent(text=text, tier_used=tier_used))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            wait = self.min_gap_seconds - (time.time() - self._last_spoken_at)
            if wait > 0:
                time.sleep(wait)
            try:
                speak(event.text)
                self._last_spoken_at = time.time()
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._stop.set()
