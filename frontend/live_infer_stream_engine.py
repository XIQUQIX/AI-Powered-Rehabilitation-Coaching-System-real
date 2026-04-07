
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from infer_stream_v2 import (
    MODEL_URL,
    PoseTCNTyped,
    download_if_missing,
    ensure_bgr,
    normalize_pose_seq,
    pick_device,
    safe_float_key,
)


@dataclass(frozen=True)
class InferRuntimeConfig:
    ckpt_path: str
    out_jsonl_path: str = "logs/infer_stream_live.jsonl"
    mediapipe_task_path: str = "models/pose_landmarker_full.task"
    window: int = 64
    stride: int = 8
    device: str = "auto"
    line_thickness: int = 2
    circle_radius: int = 2
    mist_thresh: float = 0.35
    min_exercise_p: float = 0.25
    source_fps: float = 30.0


class LiveInferStreamEngine:
    def __init__(self, config: InferRuntimeConfig):
        self.config = config
        self._load_model_and_pose_stack()
        self.Lbuf: List[np.ndarray] = []
        self.Vbuf: List[np.ndarray] = []
        self.frame_idx = 0
        self.t0 = time.perf_counter()
        self.latest_overlay: Dict[str, Any] = {"exercise": "waiting", "mistakes": [], "quality_score": None, "timestamp_s": 0.0}
        self.pose_connections = list(mp.solutions.pose.POSE_CONNECTIONS)

    def _load_model_and_pose_stack(self) -> None:
        ckpt_path = Path(self.config.ckpt_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        voc = ckpt.get("vocabs", {})
        self.ex2i = voc.get("ex2i", {})
        self.mist2i = voc.get("mist2i", {})
        self.speed2i = voc.get("speed2i", {})
        self.rom2i = voc.get("rom2i", {})
        self.height2i = voc.get("height2i", {})
        self.torso2i = voc.get("torso2i", {})
        self.dir2i = voc.get("dir2i", {})
        self.i2ex = {i: k for k, i in self.ex2i.items()}
        self.i2mist = {i: k for k, i in self.mist2i.items()}
        self.i2speed = {i: k for k, i in self.speed2i.items()}
        self.i2rom = {i: k for k, i in self.rom2i.items()} if self.rom2i else {}
        self.i2height = {i: k for k, i in self.height2i.items()} if self.height2i else {}
        self.i2torso = {i: k for k, i in self.torso2i.items()} if self.torso2i else {}
        self.i2dir = {i: k for k, i in self.dir2i.items()} if self.dir2i else {0: "none", 1: "clockwise", 2: "counterclockwise"}
        feat_dim = int(ckpt.get("feat_dim", 198))
        if feat_dim != 198:
            raise RuntimeError(f"Checkpoint feat_dim={feat_dim} but this app expects 198 features.")
        self.model = PoseTCNTyped(feat_dim=feat_dim, n_ex=max(1, len(self.ex2i)), n_mist=max(1, len(self.mist2i)), n_speed=max(1, len(self.speed2i)), hidden=256, layers=6)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.device = pick_device(self.config.device)
        self.model.to(self.device)
        model_path = download_if_missing(MODEL_URL, Path(self.config.mediapipe_task_path))
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        frame_bgr = ensure_bgr(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
            z = np.array([lm.z for lm in lms], dtype=np.float32)
            vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms], dtype=np.float32)
            L = np.concatenate([xy, z[:, None]], axis=1)
            V = vis
        else:
            L = np.full((33, 3), np.nan, dtype=np.float32)
            V = np.zeros((33,), dtype=np.float32)
        self.Lbuf.append(L); self.Vbuf.append(V)
        if len(self.Lbuf) > self.config.window:
            self.Lbuf.pop(0); self.Vbuf.pop(0)
        emitted_event = None
        if len(self.Lbuf) == self.config.window and (self.frame_idx % self.config.stride == 0):
            emitted_event = self._run_tcn_head()
            if emitted_event is not None:
                self.latest_overlay = {"exercise": emitted_event["exercise"]["name"], "mistakes": emitted_event["mistakes"], "quality_score": emitted_event["quality_score"], "timestamp_s": emitted_event["timestamp_s"]}
                self._append_jsonl(emitted_event)
        display_frame = self._draw_pose_overlay(frame_bgr, result)
        display_frame = self._draw_text_overlay(display_frame)
        self.frame_idx += 1
        return display_frame, emitted_event

    def _run_tcn_head(self) -> Dict[str, Any]:
        Lseq = np.stack(self.Lbuf, axis=0); Vseq = np.stack(self.Vbuf, axis=0)
        X = normalize_pose_seq(Lseq, Vseq)
        with torch.no_grad():
            xt = torch.from_numpy(X[None, ...]).to(self.device)
            out = self.model(xt)
            ex_prob = F.softmax(out["ex"], dim=1).cpu().numpy()[0]
            mist_prob = torch.sigmoid(out["mist"]).cpu().numpy()[0]
            no_issue_p = float(torch.sigmoid(out["no_issue"]).cpu().numpy()[0])
            dir_prob = F.softmax(out["dir"], dim=1).cpu().numpy()[0]
            speed_prob = F.softmax(out["speed"], dim=1).cpu().numpy()[0] if out["speed"].shape[1] > 1 else None
            rom_prob = F.softmax(out["rom"], dim=1).cpu().numpy()[0]
            height_prob = F.softmax(out["height"], dim=1).cpu().numpy()[0]
            torso_prob = F.softmax(out["torso"], dim=1).cpu().numpy()[0]
        ex_id = int(np.argmax(ex_prob)); ex_p = float(ex_prob[ex_id]); ex_name = self.i2ex.get(ex_id, str(ex_id))
        if ex_p < self.config.min_exercise_p: ex_name = "unknown"
        exercise = {"name": ex_name, "p": ex_p}
        mistakes = []
        if mist_prob.size > 0:
            top_idx = np.argsort(-mist_prob)[:10]
            for i in top_idx:
                p = float(mist_prob[int(i)])
                if p >= self.config.mist_thresh:
                    mistakes.append({"name": self.i2mist.get(int(i), str(int(i))), "p": p})
        speed_val = None
        if speed_prob is not None and speed_prob.size > 0:
            sid = int(np.argmax(speed_prob)); label = self.i2speed.get(sid, str(sid)); parsed = safe_float_key(label); speed_val = parsed if parsed is not None else label
        rom_id = int(np.argmax(rom_prob)); height_id = int(np.argmax(height_prob)); torso_id = int(np.argmax(torso_prob)); dir_id = int(np.argmax(dir_prob))
        rom_level = self.i2rom.get(rom_id, rom_id + 1); height_level = self.i2height.get(height_id, height_id + 1); torso_rotation = self.i2torso.get(torso_id, torso_id + 1); direction = self.i2dir.get(dir_id, {0: "none", 1: "clockwise", 2: "counterclockwise"}[dir_id])
        penalty = float(np.sort(mist_prob)[-5:].mean()) if mist_prob.size else 0.0
        quality = float(np.clip(0.60 * no_issue_p + 0.40 * (1.0 - penalty), 0.0, 1.0))
        return {"timestamp_s": float(time.perf_counter()-self.t0), "frame_index": int(self.frame_idx), "source_fps": float(self.config.source_fps), "exercise": exercise, "mistakes": mistakes, "metrics": {"speed_rps": speed_val, "rom_level": rom_level, "height_level": height_level, "torso_rotation": torso_rotation, "direction": direction, "no_obvious_issue_p": no_issue_p}, "quality_score": quality, "speak_now": 0.0}

    def _draw_pose_overlay(self, frame_bgr: np.ndarray, result: Any) -> np.ndarray:
        display = frame_bgr.copy()
        if not result.pose_landmarks:
            return display
        h, w = display.shape[:2]
        pts = []
        for lm in result.pose_landmarks[0]:
            x = int(lm.x * w); y = int(lm.y * h)
            pts.append((x,y,getattr(lm,'visibility',1.0)))
        for start, end in self.pose_connections:
            if start < len(pts) and end < len(pts):
                x1,y1,v1 = pts[start]; x2,y2,v2 = pts[end]
                if v1 > 0.2 and v2 > 0.2:
                    cv2.line(display, (x1,y1), (x2,y2), (0,255,0), self.config.line_thickness, cv2.LINE_AA)
        for x,y,v in pts:
            if v > 0.2:
                cv2.circle(display, (x,y), self.config.circle_radius, (0,255,255), -1, cv2.LINE_AA)
        return display

    def _draw_text_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        display = frame_bgr.copy()
        lines = [f"Exercise: {self.latest_overlay.get('exercise','waiting')}"]
        mistakes = self.latest_overlay.get('mistakes', []) or []
        if mistakes:
            lines.append('Mistakes: ' + ', '.join(f"{m['name']} ({m['p']:.2f})" for m in mistakes[:3]))
        else:
            lines.append('Mistakes: none above threshold')
        qs = self.latest_overlay.get('quality_score')
        if qs is not None:
            lines.append(f"Quality: {qs:.2f}")
        y=30
        for line in lines:
            cv2.putText(display, line, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(display, line, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 1, cv2.LINE_AA)
            y += 28
        return display

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        path = Path(self.config.out_jsonl_path).expanduser(); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def close(self) -> None:
        try: self.landmarker.close()
        except Exception: pass
