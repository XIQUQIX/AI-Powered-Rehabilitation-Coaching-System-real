
from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.integration.integration_layer import Config, IntegrationLayer
from src.integration.ground_truth_library import GroundTruthLibrary
from src.integration.graph import create_coaching_graph
from src.agents.progress_tracker_agent.progress_tracker_agent import ProgressTrackerAgent
from src.agents.progress_tracker_agent.rag_retriever import CoachingKnowledgeBase
from src.agents.progress_tracker_agent.schemas import PatientContext, ConditionCategory, RehabPhase, ExerciseRecord
from speech_manager import SpeechManager


@dataclass
class AppRuntimeConfig:
    session_id: str
    ground_truth_path: str = "data/ground_truth_coaching_cues.json"
    coaching_log_path: str = "logs/coaching_events_live.jsonl"
    report_chroma_dir: str = "src/agents/progress_tracker_agent/chroma_coaching_db"
    report_data_dir: str = "dataset/clean"
    anthropic_model: str = "claude-sonnet-4-20250514"
    ollama_model: str = "gemma3:4b"
    ollama_base_url: str = "http://localhost:11434"
    debug_fast: bool = True
    tts_enabled: bool = False


@dataclass
class SharedAppState:
    coaching_active: bool = False
    latest_infer_event: Optional[Dict[str, Any]] = None
    latest_coaching_delivery: Optional[Dict[str, Any]] = None
    latest_error: Optional[str] = None
    processor_status: str = "idle"
    stream_status: str = "waiting"
    frames_seen: int = 0
    inference_events_emitted: int = 0
    coaching_messages_emitted: int = 0
    last_frame_at: float = 0.0
    last_event_at: float = 0.0
    history_limit: int = 25
    coaching_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=25))
    report_text: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_coaching_active(self, active: bool) -> None:
        with self._lock: self.coaching_active = active
    def is_coaching_active(self) -> bool:
        with self._lock: return self.coaching_active
    def set_processor_status(self, status: str) -> None:
        with self._lock: self.processor_status = status
    def mark_frame(self) -> None:
        with self._lock:
            self.frames_seen += 1; self.last_frame_at = time.time(); self.stream_status = 'receiving'
    def update_infer_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.latest_infer_event = copy.deepcopy(event); self.inference_events_emitted += 1; self.last_event_at = time.time()
    def push_coaching_delivery(self, delivery: Dict[str, Any]) -> None:
        with self._lock:
            self.latest_coaching_delivery = copy.deepcopy(delivery); self.coaching_history.append(copy.deepcopy(delivery)); self.coaching_messages_emitted += 1
    def set_report_text(self, text: str) -> None:
        with self._lock: self.report_text = text
    def clear_transcript(self) -> None:
        with self._lock:
            self.latest_coaching_delivery = None; self.coaching_history.clear(); self.report_text = ''
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"coaching_active": self.coaching_active, "latest_infer_event": copy.deepcopy(self.latest_infer_event), "latest_coaching_delivery": copy.deepcopy(self.latest_coaching_delivery), "latest_error": self.latest_error, "processor_status": self.processor_status, "stream_status": self.stream_status, "frames_seen": self.frames_seen, "inference_events_emitted": self.inference_events_emitted, "coaching_messages_emitted": self.coaching_messages_emitted, "coaching_history": list(copy.deepcopy(self.coaching_history)), "report_text": self.report_text}


class RehabFullAppWrapper:
    def __init__(self, config: AppRuntimeConfig):
        self.config = config
        self.lock = threading.Lock()
        self.ground_truth_library = GroundTruthLibrary(config.ground_truth_path)
        self.integration_layer = IntegrationLayer(session_id=config.session_id, config=self._build_config(config.debug_fast), gt_library=self.ground_truth_library)
        self.integration_layer.cache.populate_defaults()
        self.graph = create_coaching_graph()
        self.session_events: List[Dict[str, Any]] = []
        self.coaching_deliveries: List[Dict[str, Any]] = []
        self.speech = SpeechManager(enabled=config.tts_enabled, min_gap_seconds=5.0)
        self.report_agent: Optional[ProgressTrackerAgent] = None

    def set_tts_enabled(self, enabled: bool) -> None:
        self.speech.set_enabled(enabled)

    def process_inference_event(self, infer_event: Dict[str, Any], patient_context_note: str, patient_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self.lock:
            coaching_event = self.integration_layer.process_frame(infer_event)
            if coaching_event is None:
                return None
            initial_state = {
                "coaching_event": coaching_event,
                "session_id": self.integration_layer.session_id,
                "coaching_history": self.integration_layer.coaching_history,
                "cache": self.integration_layer.cache,
                "ground_truth_library": self.ground_truth_library,
                "tier": coaching_event.get("tier"),
                "cache_key": coaching_event.get("cache_key"),
                "routing_reason": coaching_event.get("routing_reason"),
                "patient_profile": patient_profile,
                "patient_context_note": patient_context_note,
            }
            final_state = self.graph.invoke(initial_state)
            self.integration_layer.record_coaching_complete(coaching_event, final_state["feedback_audio"], final_state["tier_used"])
            delivery = {
                "event_id": coaching_event["event_id"],
                "timestamp": coaching_event["timestamp"],
                "message": final_state["feedback_audio"],
                "tier_used": final_state["tier_used"],
                "timing": final_state.get("delivery_timing", "rep_end"),
                "latency_ms": final_state.get("latency_ms", 0.0),
                "used_fallback": final_state.get("used_fallback", False),
                "fallback_source": final_state.get("fallback_source"),
                "coaching_event": coaching_event,
            }
            self.session_events.append(copy.deepcopy(coaching_event))
            self.coaching_deliveries.append(copy.deepcopy(delivery))
            self._append_jsonl(self.config.coaching_log_path, delivery)
            self.speech.enqueue(delivery["message"], delivery["tier_used"])
            return delivery

    def generate_report_text(self, patient_context_note: str, patient_profile: Dict[str, Any], force_rebuild: bool = False) -> str:
        with self.lock:
            if not self.session_events:
                return "No coaching events were generated in this session, so there is no report yet."
            if self.report_agent is None:
                kb = CoachingKnowledgeBase(data_dir=self.config.report_data_dir, persist_dir=self.config.report_chroma_dir).load_or_build(force_rebuild=force_rebuild)
                self.report_agent = ProgressTrackerAgent(knowledge_base=kb, ollama_model=self.config.ollama_model, ollama_base_url=self.config.ollama_base_url, retrieval_k=3, enable_polish=True, verbose=True)
            recent = []
            for event in self.session_events[-5:]:
                ex = event.get("exercise", {}).get("name", "Exercise").title()
                quality = event.get("quality_score", 0.7)
                mistake = event.get("mistake", {}).get("type", "form issue")
                recent.append(ExerciseRecord(name=ex, completed=quality >= 0.4, difficulty_feedback=f"issue: {mistake}"))
            ctx = PatientContext(
                patient_id=self.integration_layer.session_id,
                condition=patient_profile.get("condition", "general rehabilitation"),
                condition_category=ConditionCategory.GENERAL_MSK,
                rehab_phase=RehabPhase.EARLY,
                pain_level=int(patient_profile.get("pain_level", 3)),
                weeks_into_rehab=int(patient_profile.get("weeks_into_rehab", 1)),
                recent_exercises=recent,
                patient_message=patient_context_note or "Generate a concise progress summary for this rehab session based on the observed coaching events.",
                age=patient_profile.get("age") if patient_profile.get("age") not in ('', None) else None,
                goals=patient_profile.get("goals", ""),
            )
            report = self.report_agent.generate_progress_report(ctx)
            return report.coaching_feedback

    def reset_session(self) -> None:
        with self.lock:
            self.integration_layer.reset_session(); self.session_events.clear(); self.coaching_deliveries.clear()

    @staticmethod
    def _build_config(debug_fast: bool) -> Config:
        config = Config()
        if debug_fast:
            config.WINDOW_SIZE_FRAMES = 30; config.MIN_FRAMES = 10; config.MIN_PERSISTENCE_RATE = 0.15; config.MIN_CONFIDENCE = 0.25; config.MIN_DURATION_SECONDS = 0.5; config.MIN_COACHING_INTERVAL = 2; config.RE_COACHING_THRESHOLD = 5
        return config

    @staticmethod
    def _append_jsonl(path_str: str, payload: Dict[str, Any]) -> None:
        path = Path(path_str).expanduser(); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def generate_session_id(prefix: str = 'streamlit_session') -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
