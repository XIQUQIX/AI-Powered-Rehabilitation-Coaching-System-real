
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import queue
import threading
import time
import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

sys.path.insert(0, str(Path(__file__).parent))

from app_wrapper import AppRuntimeConfig, RehabFullAppWrapper, SharedAppState, generate_session_id
from live_infer_stream_engine import InferRuntimeConfig, LiveInferStreamEngine


def load_env_file_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        return

load_env_file_if_available()
logging.getLogger('streamlit_webrtc').setLevel(logging.WARNING)
logging.getLogger('aioice').setLevel(logging.WARNING)

st.set_page_config(page_title='AI Rehab Coaching — Full Local App', layout='wide')


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, '') else default



DEFAULTS = {
    'ckpt_path': env_or_default('REHAB_CKPT_PATH', ''),
    'ground_truth_path': env_or_default('REHAB_GROUND_TRUTH_PATH', 'data/ground_truth_coaching_cues.json'),
    'infer_log': env_or_default('REHAB_INFER_LOG', 'logs/infer_stream_live.jsonl'),
    'coaching_log': env_or_default('REHAB_COACHING_LOG', 'logs/coaching_events_live.jsonl'),
    'report_data_dir': env_or_default('REHAB_REPORT_DATA_DIR', 'dataset/clean'),
    'report_chroma_dir': env_or_default('REHAB_REPORT_CHROMA_DIR', 'src/agents/progress_tracker_agent/chroma_coaching_db'),
    'device': env_or_default('REHAB_DEVICE', 'auto'),
    'debug_fast': env_or_default('REHAB_DEBUG_FAST', 'true').lower() == 'true',
    'window': int(env_or_default('REHAB_WINDOW', '64')),
    'stride': int(env_or_default('REHAB_STRIDE', '8')),
    'mist_thresh': float(env_or_default('REHAB_MIST_THRESH', '0.35')),
    'min_exercise_p': float(env_or_default('REHAB_MIN_EXERCISE_P', '0.25')),
    'mediapipe_task_path': env_or_default('REHAB_MEDIAPIPE_TASK_PATH', 'models/pose_landmarker_full.task'),
    'anthropic_model': env_or_default('REHAB_ANTHROPIC_MODEL', 'claude-sonnet-4-20250514'),
    'ollama_model': env_or_default('REHAB_OLLAMA_MODEL', 'gemma3:4b'),
}

def stable_signature(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()

if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedAppState(history_limit=25)

if "backend" not in st.session_state:
    st.session_state.backend = None

if "runtime_signature" not in st.session_state:
    st.session_state.runtime_signature = None

if "runtime_payload" not in st.session_state:
    st.session_state.runtime_payload = None

if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()

if "patient_context" not in st.session_state:
    st.session_state.patient_context = ""

if "coaching_active" not in st.session_state:
    st.session_state.coaching_active = False

if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False

if "session_report" not in st.session_state:
    st.session_state.session_report = ""

st.title('AI-Powered Rehabilitation Coaching — Local Full App')
st.caption('Continuous browser video + local checkpoint inference + Anthropic live coaching + Ollama report generation + optional TTS.')

patient_context = st.text_area('Patient context for this session', value=st.session_state.patient_context, help='Optional note from the patient. This is injected as extra context for live coaching and the final session report, then cleared when the session resets.')
st.session_state.patient_context = patient_context

with st.expander('Optional patient profile', expanded=False):
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        patient_name = st.text_input('Patient name', value='Patient')
        patient_age = st.text_input('Age', value='')
    with colp2:
        condition = st.text_input('Condition', value='general rehabilitation')
        goals = st.text_input('Goal', value='')
    with colp3:
        pain_level = st.number_input('Pain level (0-10)', min_value=0, max_value=10, value=3)
        weeks_into_rehab = st.number_input('Weeks into rehab', min_value=0, max_value=104, value=1)

with st.sidebar:
    st.header('Runtime settings')
    ckpt_path = st.text_input('Local checkpoint path', value=DEFAULTS['ckpt_path'])
    ground_truth_path = st.text_input('Ground-truth JSON path', value=DEFAULTS['ground_truth_path'])
    infer_log = st.text_input('Infer JSONL log path', value=DEFAULTS['infer_log'])
    coaching_log = st.text_input('Coaching JSONL log path', value=DEFAULTS['coaching_log'])
    report_data_dir = st.text_input('Report corpus dir', value=DEFAULTS['report_data_dir'])
    report_chroma_dir = st.text_input('Report Chroma dir', value=DEFAULTS['report_chroma_dir'])
    device = st.selectbox('Torch device', options=['auto','cpu','mps'], index=['auto','cpu','mps'].index(DEFAULTS['device']) if DEFAULTS['device'] in {'auto','cpu','mps'} else 0)
    debug_fast = st.checkbox('Use debug-fast integration thresholds', value=DEFAULTS['debug_fast'])
    window = int(st.number_input('Inference window', min_value=8, max_value=256, value=int(DEFAULTS['window']), step=8))
    stride = int(st.number_input('Inference stride', min_value=1, max_value=64, value=int(DEFAULTS['stride']), step=1))
    mist_thresh = float(st.slider('Mistake threshold', min_value=0.05, max_value=0.95, value=float(DEFAULTS['mist_thresh']), step=0.05))
    min_exercise_p = float(st.slider('Minimum exercise probability', min_value=0.05, max_value=0.95, value=float(DEFAULTS['min_exercise_p']), step=0.05))
    mediapipe_task_path = st.text_input('MediaPipe task file path', value=DEFAULTS['mediapipe_task_path'])
    anthropic_model = st.text_input('Anthropic model', value=DEFAULTS['anthropic_model'])
    ollama_model = st.text_input('Ollama model', value=DEFAULTS['ollama_model'])
    tts_enabled = st.checkbox('Enable spoken feedback (Tier 2/Tier 3 only)', value=st.session_state.tts_enabled)
    st.session_state.tts_enabled = tts_enabled

runtime_payload = {'session_id': st.session_state.session_id, 'ckpt_path': ckpt_path, 'ground_truth_path': ground_truth_path, 'infer_log': infer_log, 'coaching_log': coaching_log, 'report_data_dir': report_data_dir, 'report_chroma_dir': report_chroma_dir, 'device': device, 'debug_fast': debug_fast, 'window': window, 'stride': stride, 'mist_thresh': mist_thresh, 'min_exercise_p': min_exercise_p, 'mediapipe_task_path': mediapipe_task_path, 'anthropic_model': anthropic_model, 'ollama_model': ollama_model, 'tts_enabled': tts_enabled}
current_signature = stable_signature(runtime_payload)
if st.session_state.runtime_signature != current_signature:
    st.session_state.backend = RehabFullAppWrapper(AppRuntimeConfig(session_id=runtime_payload['session_id'], ground_truth_path=ground_truth_path, coaching_log_path=coaching_log, report_chroma_dir=report_chroma_dir, report_data_dir=report_data_dir, anthropic_model=anthropic_model, ollama_model=ollama_model, debug_fast=debug_fast, tts_enabled=tts_enabled))
    st.session_state.shared_state = SharedAppState(history_limit=25)
    st.session_state.runtime_signature = current_signature

shared_state: SharedAppState = st.session_state.shared_state
backend: RehabFullAppWrapper = st.session_state.backend
backend.set_tts_enabled(tts_enabled)
checkpoint_exists = bool(ckpt_path) and Path(ckpt_path).expanduser().exists()

patient_profile = {'name': patient_name, 'age': int(patient_age) if str(patient_age).strip().isdigit() else None, 'condition': condition, 'goals': goals, 'pain_level': int(pain_level), 'weeks_into_rehab': int(weeks_into_rehab), 'known_limitations': [], 'past_injuries': [], 'preferences': {'coaching_style': 'encouraging', 'audio_enabled': tts_enabled, 'detailed_explanations': False}}

infer_cfg = InferRuntimeConfig(ckpt_path=ckpt_path, out_jsonl_path=infer_log, mediapipe_task_path=mediapipe_task_path, window=window, stride=stride, device=device, mist_thresh=mist_thresh, min_exercise_p=min_exercise_p)

def make_video_processor_factory(
    runtime_cfg: InferRuntimeConfig,
    app_state: SharedAppState,
    local_backend: RehabFullAppWrapper,
    patient_context_text: str,
    patient_profile_data: dict,
):
    class RehabVideoProcessor:
        def __init__(self) -> None:
            self.engine = None
            self.init_error = None
            self.event_queue: queue.Queue = queue.Queue(maxsize=2)
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._coaching_worker, daemon=True)
            self.worker.start()
            app_state.set_processor_status('initializing')

        def _ensure_engine(self) -> None:
            if self.engine is None:
                self.engine = LiveInferStreamEngine(runtime_cfg)
                app_state.set_processor_status('running')

        def _enqueue_infer_event(self, infer_event: dict) -> None:
            try:
                self.event_queue.put_nowait(infer_event)
            except queue.Full:
                try:
                    _ = self.event_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.event_queue.put_nowait(infer_event)
                except queue.Full:
                    pass

        def _coaching_worker(self) -> None:
            while not self.stop_event.is_set():
                try:
                    infer_event = self.event_queue.get(timeout=0.25)
                except queue.Empty:
                    continue

                try:
                    if app_state.is_coaching_active():
                        delivery = local_backend.process_inference_event(
                            infer_event,
                            patient_context_text,
                            patient_profile_data,
                        )
                        if delivery is not None:
                            app_state.push_coaching_delivery(delivery)
                except Exception as exc:
                    app_state.set_processor_status(f'coaching_error: {exc}')
                finally:
                    try:
                        self.event_queue.task_done()
                    except Exception:
                        pass

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format='bgr24')
            app_state.mark_frame()

            if self.init_error is not None:
                return self._error_frame(img, self.init_error)

            try:
                self._ensure_engine()
                processed, infer_event = self.engine.process_frame(img)

                if infer_event is not None:
                    app_state.update_infer_event(infer_event)
                    if app_state.is_coaching_active():
                        self._enqueue_infer_event(infer_event)

                return av.VideoFrame.from_ndarray(processed, format='bgr24')

            except Exception as exc:
                self.init_error = f'Processor error: {exc}'
                return self._error_frame(img, self.init_error)

        def _error_frame(self, frame_bgr, message: str) -> av.VideoFrame:
            app_state.set_processor_status('error')
            error_img = frame_bgr.copy()
            cv2.putText(
                error_img,
                message[:90],
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return av.VideoFrame.from_ndarray(error_img, format='bgr24')

        def __del__(self) -> None:
            try:
                self.stop_event.set()
            except Exception:
                pass
            try:
                if self.engine is not None:
                    self.engine.close()
            except Exception:
                pass

    return RehabVideoProcessor

c1,c2,c3,c4 = st.columns(4)
with c1:
    if st.button('Start Coaching', use_container_width=True, disabled=not checkpoint_exists): shared_state.set_coaching_active(True)
with c2:
    if st.button('Stop Coaching', use_container_width=True): shared_state.set_coaching_active(False)
with c3:
    if st.button('Generate Report', use_container_width=True):
        report_text = backend.generate_report_text(st.session_state.patient_context, patient_profile, force_rebuild=False)
        shared_state.set_report_text(report_text)
with c4:
    if st.button('Reset Session', use_container_width=True):
        backend.reset_session(); shared_state.clear_transcript(); st.session_state.patient_context = ''; st.session_state.session_id = generate_session_id(); st.session_state.runtime_signature = None; st.rerun()

if not checkpoint_exists: st.warning('Set a valid local checkpoint path in the sidebar before starting coaching.')
st.info('Use the camera component START/STOP controls to open or close the webcam stream. The Start Coaching / Stop Coaching buttons only toggle feedback generation.')

ctx = webrtc_streamer(
    key='rehab-full-local',
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={'video': True, 'audio': False},
    async_processing=True,
    video_processor_factory=make_video_processor_factory(
        infer_cfg,
        shared_state,
        backend,
        patient_context,
        patient_profile,
    ),
)

@st.fragment(run_every=0.5)
def render_live_panels():
    snap = shared_state.snapshot()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Live coaching')
        latest = snap.get('latest_coaching_delivery')
        if latest:
            st.markdown(f"**Tier:** {latest['tier_used']}")
            st.write(latest['message'])
        else:
            st.write('No coaching cue yet.')

        st.subheader('Session transcript')
        history = snap.get('coaching_history', [])
        if history:
            for item in reversed(history[-25:]):
                st.markdown(f"**{item['tier_used']}** — {item['message']}")
        else:
            st.write('No transcript entries yet.')

    with col2:
        st.subheader('Status')
        st.json({
            'coaching_active': snap['coaching_active'],
            'processor_status': snap['processor_status'],
            'frames_seen': snap['frames_seen'],
            'inference_events_emitted': snap['inference_events_emitted'],
            'coaching_messages_emitted': snap['coaching_messages_emitted'],
        })

        st.subheader('Latest infer event')
        st.json(snap.get('latest_infer_event') or {})

        st.subheader('Session report')
        st.write(snap.get('report_text') or 'Generate a report at the end of a session to view it here.')

render_live_panels()
