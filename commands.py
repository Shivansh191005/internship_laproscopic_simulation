"""
Command bus.

Voice and keyboard both push commands here; the controllers read state from
here. Keeping arbitration in one place means the servo never has to know where
an instruction came from, and there is exactly one definition of what each
command does.

Safety note: voice commands can bias the scope and nudge the suction arm, but
they cannot override the critical-structure virtual fixture. A spoken
instruction is an input to the controller, not a bypass of it.
"""
import threading
import time

import numpy as np

import config as cfg

# Spoken phrase -> canonical command. Small on purpose: short, acoustically
# distinct words are recognised reliably in a noisy OR. Every phrase must be
# said after the wake word (cfg.VOICE_WAKE_WORD, "ana"), e.g.
# "ana zoom in"; only "stop" works on its own, at any time.
GRAMMAR = {
    # camera arm (endoscope)
    "zoom in": "ZOOM_IN",
    "closer": "ZOOM_IN",
    "zoom out": "ZOOM_OUT",
    "wider": "ZOOM_OUT",
    "left": "PAN_LEFT",
    "right": "PAN_RIGHT",
    "up": "PAN_UP",
    "down": "PAN_DOWN",
    "lower": "PAN_DOWN",           # recognised more reliably than "down"
    "center": "CENTER",
    "centre": "CENTER",
    "follow": "CAM_FOLLOW",
    "hold": "CAM_HOLD",
    "freeze": "CAM_HOLD",          # clearer than "hold" (sounds like "go")
    "home": "HOME",
    "reset": "HOME",
    "light up": "LIGHT_UP",
    "brighter": "LIGHT_UP",
    "light down": "LIGHT_DOWN",
    "darker": "LIGHT_DOWN",
    "dimmer": "LIGHT_DOWN",
    "light out": "LIGHT_DOWN",     # how "light down" is often misheard
    # instrument arm
    "tool follow": "TOOL_FOLLOW",
    "tool hold": "TOOL_HOLD",
    # record keeping
    "picture": "SNAPSHOT",         # ("snapshot" ends in "...shot" ~ "stop")
    "mark": "MARK",
}
STOP_WORD = "stop"                 # no wake word needed: always available

KEYMAP = {
    ord("f"): "CAM_FOLLOW", ord("h"): "CAM_HOLD", ord("c"): "CENTER",
    ord("g"): "HOME",
    ord("="): "ZOOM_IN", ord("+"): "ZOOM_IN",
    ord("-"): "ZOOM_OUT", ord("_"): "ZOOM_OUT",
    ord("a"): "PAN_LEFT", ord("d"): "PAN_RIGHT",
    ord("w"): "PAN_UP", ord("x"): "PAN_DOWN",
    ord("1"): "TOOL_FOLLOW", ord("2"): "TOOL_HOLD", ord("3"): "TOOL_HOLD",
    ord("z"): "STOP", ord("m"): "MARK",
    ord("l"): "LIGHT_UP", ord("k"): "LIGHT_DOWN",
}

# commands handled by the main loop as one-off events (camera view etc.)
EVENTS = {"ZOOM_IN", "ZOOM_OUT", "PAN_LEFT", "PAN_RIGHT", "PAN_UP",
          "PAN_DOWN", "CENTER", "CAM_FOLLOW", "CAM_HOLD", "HOME", "STOP",
          "SNAPSHOT", "MARK", "LIGHT_UP", "LIGHT_DOWN"}


def wake_words():
    w = getattr(cfg, "VOICE_WAKE_WORD", "") or ""
    if not w:
        return []
    return [w] + list(getattr(cfg, "VOICE_WAKE_ALIASES", []) or [])


def vosk_phrases():
    """Exact sentences the recogniser may output (restricted grammar)."""
    ws = wake_words() or [""]
    out = [f"{w} {p}".strip() for w in ws for p in GRAMMAR] + [STOP_WORD]
    return sorted(set(out))


class CommandBus:
    def __init__(self):
        self._lock = threading.Lock()
        self.mode = "FOLLOW"              # FOLLOW | HOLD
        self.instrument = "FOLLOW"        # arm 2: FOLLOW | HOLD
        self.bias = np.zeros(2)           # operator pan offset, normalised
        self.zoom_bias = 0.0              # added to TARGET_AREA_FRAC
        self.last = ""
        self.last_t = 0.0
        self.log = []
        self._events = []

    def push(self, command, source="voice"):
        if command is None:
            return
        with self._lock:
            self.last = f"{command} ({source})"
            self.last_t = time.time()
            self.log.append((time.time(), command, source))

            if command == "FOLLOW":
                self.mode = "FOLLOW"
                self.bias[:] = 0.0
                if cfg.CAMERA_MODE == "stable":   # camera is fixed: the
                    self.instrument = "FOLLOW"    # words drive arm 2
            elif command == "HOLD":
                self.mode = "HOLD"
                if cfg.CAMERA_MODE == "stable":
                    self.instrument = "HOLD"
            elif command == "CENTER":
                self.mode = "FOLLOW"
                self.bias[:] = 0.0
                self.zoom_bias = 0.0
            elif command == "ZOOM_IN":
                self.zoom_bias = float(np.clip(self.zoom_bias + 0.015,
                                               -0.04, 0.06))
            elif command == "ZOOM_OUT":
                self.zoom_bias = float(np.clip(self.zoom_bias - 0.015,
                                               -0.04, 0.06))
            elif command == "PAN_LEFT":
                self.bias[0] = float(np.clip(self.bias[0] - cfg.PAN_NUDGE,
                                             -0.6, 0.6))
            elif command == "PAN_RIGHT":
                self.bias[0] = float(np.clip(self.bias[0] + cfg.PAN_NUDGE,
                                             -0.6, 0.6))
            elif command == "PAN_UP":
                self.bias[1] = float(np.clip(self.bias[1] - cfg.PAN_NUDGE,
                                             -0.6, 0.6))
            elif command == "PAN_DOWN":
                self.bias[1] = float(np.clip(self.bias[1] + cfg.PAN_NUDGE,
                                             -0.6, 0.6))
            elif command == "TOOL_FOLLOW":
                self.instrument = "FOLLOW"
            elif command == "TOOL_HOLD":
                self.instrument = "HOLD"
            elif command == "STOP":           # freeze BOTH arms
                self.mode = "HOLD"
                self.instrument = "HOLD"
            if command in EVENTS:
                self._events.append((command, source))

    def pop_events(self):
        """One-off commands for the main loop (camera view, snapshot...)."""
        with self._lock:
            ev, self._events = self._events, []
        return ev

    def snapshot(self):
        with self._lock:
            return {"mode": self.mode, "instrument": self.instrument,
                    "suction": "ON" if self.instrument == "FOLLOW" else "OFF",
                    "bias": self.bias.copy(), "zoom_bias": self.zoom_bias,
                    "last": self.last, "age": time.time() - self.last_t}

    def key(self, k):
        if k in KEYMAP:
            self.push(KEYMAP[k], source="key")
            return True
        return False


def match_phrase(text, wake=None):
    """
    Text heard -> command, or None.
      * "stop" anywhere -> STOP (no wake word needed);
      * otherwise the wake word must be present and the command must follow
        it ("ana zoom in"); longest phrase wins ("tool follow" > "follow").
    wake=None uses cfg.VOICE_WAKE_WORD; wake="" disables the wake word.
    """
    t = " ".join((text or "").lower().replace("-", " ").split())
    if not t:
        return None
    words = t.split()
    if STOP_WORD in words:
        return "STOP"
    ws = wake_words() if wake is None else ([wake] if wake else [])
    if ws:
        pos = [i for i, x in enumerate(words) if x in ws]
        if not pos:
            return None
        t = " ".join(words[pos[-1] + 1:])          # what follows the wake word
    padded = f" {t} "
    for phrase in sorted(GRAMMAR, key=len, reverse=True):
        if f" {phrase} " in padded:
            return GRAMMAR[phrase]
    return None
