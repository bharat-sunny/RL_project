"""Record the arm while a policy drives it, with live telemetry burned in.

Raw footage of an arm moving does not show what the agent is doing.  What makes
a demo self-explanatory is the state the policy is acting on, drawn on the frame
as it happens: which target is active, how far the end effector still is, which
control step this is, and whether the trial ended inside tolerance.

Frames are captured on a background thread at a steady rate rather than once per
control step.  A control step takes roughly 400 ms, so a frame-per-step video
would be four frames a second and unwatchable; the thread keeps the footage
smooth while the trial loop simply publishes its latest telemetry.

IMPORTANT — the camera is not part of the control loop.  The policy observes
end-effector position and velocity reported by the arm's own encoders, exactly
as in simulation, and never sees a pixel.  The wrist camera view is illustrative
only; captions must not imply the agent is doing visual servoing.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

# Drawn in the same ink as the report figures so the video and the deck match.
INK = (11, 11, 11)
INK_SECONDARY = (78, 81, 82)
SURFACE = (251, 252, 252)
ACCENT = (214, 120, 42)       # BGR of #2a78d6
ACCENT_WARM = (52, 104, 235)  # BGR of #eb6834
GOOD = (122, 175, 27)         # BGR of #1baf7a


class ArmRecorder:
    """Capture annotated video of a trial run on a background thread."""

    def __init__(self, output: Path, device: int = 4, fps: int = 15,
                 width: int = 1280, height: int = 720, label: str = "") -> None:
        import cv2

        self.cv2 = cv2
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.label = label

        # The RealSense was found with auto white balance *disabled* and pinned to
        # 4600 K, which does not match this room and tints every frame pink.  It
        # is set here rather than left to a manual v4l2 call, so a recording is
        # never silently miscoloured by whatever state the camera was left in.
        self._configure_camera(device)

        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open camera /dev/video{device}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Auto exposure and white balance need frames to converge; without this
        # the opening seconds of every clip are visibly off-colour.
        settle_until = time.time() + 3.0
        while time.time() < settle_until:
            self.capture.read()

        # Use whatever the camera actually gave us, not what we asked for.
        ok, frame = self.capture.read()
        if not ok:
            self.capture.release()
            raise RuntimeError(f"camera /dev/video{device} opened but returned no frame")
        self.height, self.width = frame.shape[:2]

        self.writer = cv2.VideoWriter(
            str(self.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (self.width, self.height))
        if not self.writer.isOpened():
            self.capture.release()
            raise RuntimeError(f"could not open a writer for {self.output}")

        self._telemetry: dict = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.frames_written = 0

    @staticmethod
    def _configure_camera(device: int) -> None:
        """Turn on automatic white balance and exposure via v4l2.

        OpenCV's ``CAP_PROP_AUTO_WB`` is unreliable across UVC drivers, so the
        control is set directly.  Failure is non-fatal — a tinted recording is
        worth having, an aborted one is not.
        """
        import subprocess

        for control in ("white_balance_automatic=1", "auto_exposure=3"):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", f"/dev/video{device}", "--set-ctrl", control],
                    check=False, capture_output=True, timeout=5,
                )
            except Exception:
                pass

    # ------------------------------------------------------------- telemetry

    def update(self, **fields) -> None:
        """Publish the latest trial state; the capture thread picks it up."""
        with self._lock:
            self._telemetry.update(fields)

    # ---------------------------------------------------------------- drawing

    def _annotate(self, frame: np.ndarray, telemetry: dict) -> np.ndarray:
        """Draw the telemetry panel.

        The panel sits at the *bottom*: the arm occupies the upper half of the
        frame, and an overlay across the top hides the very thing the video
        exists to show.  Text is ASCII only — OpenCV's built-in Hershey fonts
        have no glyphs beyond it and silently render '?' for anything else.
        """
        cv2 = self.cv2
        font = cv2.FONT_HERSHEY_SIMPLEX

        panel_h = 118
        top = self.height - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, top), (self.width, self.height), SURFACE, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, top), (self.width, top), (228, 231, 232), 2)

        title = telemetry.get("label", self.label)
        cv2.putText(frame, title, (22, top + 30), font, 0.66, INK, 2, cv2.LINE_AA)

        trial = telemetry.get("trial")
        total = telemetry.get("n_trials")
        if trial is not None:
            cv2.putText(frame, f"trial {trial}/{total}    step {telemetry.get('step', 0)}",
                        (22, top + 58), font, 0.54, INK_SECONDARY, 1, cv2.LINE_AA)

        distance = telemetry.get("distance_mm")
        tolerance = telemetry.get("tolerance_mm", 10.0)
        if distance is not None:
            colour = GOOD if distance < tolerance else ACCENT
            cv2.putText(frame, f"distance to target  {distance:5.1f} mm",
                        (22, top + 92), font, 0.62, colour, 2, cv2.LINE_AA)

            # A bar makes the approach readable at a glance; full width = 60 mm out.
            bar_x, bar_w, bar_y = 470, 360, top + 76
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18),
                          (232, 231, 228), -1)
            filled = int(bar_w * min(1.0, distance / 60.0))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 18), colour, -1)
            tol_x = bar_x + int(bar_w * (tolerance / 60.0))
            cv2.line(frame, (tol_x, bar_y - 6), (tol_x, bar_y + 24), INK_SECONDARY, 2)
            cv2.putText(frame, f"{tolerance:.0f} mm tolerance", (tol_x + 8, bar_y - 10),
                        font, 0.42, INK_SECONDARY, 1, cv2.LINE_AA)

        outcome = telemetry.get("outcome")
        if outcome:
            colour = GOOD if outcome == "SUCCESS" else ACCENT_WARM
            cv2.putText(frame, outcome, (self.width - 200, top + 92), font, 0.85,
                        colour, 2, cv2.LINE_AA)

        # State the control interface, so nobody reads this as visual servoing.
        cv2.putText(frame,
                    "policy input: end-effector position + velocity (no vision)",
                    (22, top - 16), font, 0.5, INK_SECONDARY, 1, cv2.LINE_AA)

        self._draw_schematic(frame, telemetry)
        return frame

    # -------------------------------------------------------------- schematic

    def _draw_schematic(self, frame: np.ndarray, telemetry: dict) -> None:
        """Draw a live map of target and end effector.

        Without this the video is uninterpretable: the goal is a coordinate in
        empty space, so a viewer sees an arm move and has no way to tell where it
        was trying to go or whether it got there.  Marking the table would not
        fix it either, because targets vary in height as well as position.

        Two small panels — seen from above and from the side — carry the whole
        3-D relationship.  The workspace outline gives scale, the ring is the
        target with its tolerance, and the dot is where the arm actually is.
        Everything is computed from the same telemetry the policy acts on, so the
        schematic cannot disagree with the run it depicts.
        """
        cv2 = self.cv2
        ee = telemetry.get("ee_mm")
        goal = telemetry.get("goal_mm")
        box = telemetry.get("box")            # (centre, half_extent) in mm
        if ee is None or goal is None or box is None:
            return

        centre, half = np.asarray(box[0], float), np.asarray(box[1], float)
        ee, goal = np.asarray(ee, float), np.asarray(goal, float)
        tolerance = telemetry.get("tolerance_mm", 10.0)

        pad, size, gap = 16, 150, 14
        origin_x = self.width - (size * 2 + gap + pad)
        origin_y = pad
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Panels are drawn with a margin so a point at the workspace edge is not
        # clipped to the border and made to look like a different position.
        margin = 0.22

        def panel(index: int, axis_a: int, axis_b: int, title: str, flip_b: bool):
            x0 = origin_x + index * (size + gap)
            y0 = origin_y

            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + size, y0 + size), SURFACE, -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (222, 224, 225), 1)
            cv2.putText(frame, title, (x0 + 8, y0 + 18), font, 0.42,
                        INK_SECONDARY, 1, cv2.LINE_AA)

            span_a = half[axis_a] * 2 * (1 + margin * 2)
            span_b = half[axis_b] * 2 * (1 + margin * 2)

            def to_px(point):
                u = (point[axis_a] - centre[axis_a]) / span_a + 0.5
                v = (point[axis_b] - centre[axis_b]) / span_b + 0.5
                if flip_b:
                    v = 1.0 - v
                return (int(x0 + u * size), int(y0 + v * size))

            # Workspace outline: the region targets are drawn from.
            c1 = to_px(centre - half)
            c2 = to_px(centre + half)
            cv2.rectangle(frame, (min(c1[0], c2[0]), min(c1[1], c2[1])),
                          (max(c1[0], c2[0]), max(c1[1], c2[1])),
                          (206, 209, 210), 1)

            gp, ep = to_px(goal), to_px(ee)
            reached = float(np.linalg.norm(ee - goal)) < tolerance
            colour = GOOD if reached else ACCENT

            cv2.line(frame, ep, gp, (214, 216, 217), 1, cv2.LINE_AA)

            # Target: a ring sized to the success tolerance, so "close enough" is
            # visible rather than something the viewer has to take on trust.
            radius = max(4, int(size * tolerance / span_a))
            cv2.circle(frame, gp, radius, colour, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, gp, colour, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)

            # End effector.
            cv2.circle(frame, ep, 6, INK, -1, cv2.LINE_AA)
            cv2.circle(frame, ep, 6, SURFACE, 1, cv2.LINE_AA)

        panel(0, 0, 1, "seen from above", flip_b=False)
        panel(1, 0, 2, "seen from the side", flip_b=True)

        legend_y = origin_y + size + 18
        cv2.circle(frame, (origin_x + 8, legend_y - 4), 5, INK, -1, cv2.LINE_AA)
        cv2.putText(frame, "arm", (origin_x + 20, legend_y), font, 0.4,
                    INK_SECONDARY, 1, cv2.LINE_AA)
        cv2.circle(frame, (origin_x + 78, legend_y - 4), 6, ACCENT, 2, cv2.LINE_AA)
        cv2.putText(frame, "target + tolerance", (origin_x + 92, legend_y), font, 0.4,
                    INK_SECONDARY, 1, cv2.LINE_AA)

    # ----------------------------------------------------------------- thread

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while self._running:
            start = time.perf_counter()
            ok, frame = self.capture.read()
            if ok:
                with self._lock:
                    telemetry = dict(self._telemetry)
                self.writer.write(self._annotate(frame, telemetry))
                self.frames_written += 1
            elapsed = time.perf_counter() - start
            if elapsed < period:
                time.sleep(period - elapsed)

    def start(self) -> "ArmRecorder":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Path:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.writer.release()
        self.capture.release()
        return self.output

    def __enter__(self) -> "ArmRecorder":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
