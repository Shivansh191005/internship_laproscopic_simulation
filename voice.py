"""
Voice command input.

Uses Vosk: offline, small, runs on CPU, and does not compete with YOLO for your
6 GB of VRAM. Cloud speech APIs add 300-800 ms of round trip, which is the wrong
trade for short imperative commands.

Setup:
    pip install vosk sounddevice
    Download vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models
    Unzip to  models/vosk-small-en/

If Vosk or the model is missing, this degrades to keyboard-only and says so.
Nothing else in the system changes — commands.CommandBus does not care where a
command came from.

A restricted grammar is passed to the recogniser so it only listens for the
command words. This matters: an open vocabulary in a noisy room produces
constant false triggers, and a false 'zoom in' during dissection is a real
distraction.
"""
import json
import os
import queue
import threading

import time

from commands import match_phrase, vosk_phrases
import config as cfg


class VoiceListener(threading.Thread):
    def __init__(self, bus, model_dir=None):
        super().__init__(daemon=True)
        self.bus = bus
        self.model_dir = model_dir or cfg.VOSK_MODEL_DIR
        # NOT "_stop": threading.Thread uses that name internally (join()
        # then fails with "Event object is not callable")
        self._stop_evt = threading.Event()
        self.heard = ""
        self._last_cmd = (None, 0.0)
        self.available = False
        self.status = "not started"
        self._q = queue.Queue()
        self._rec = None
        self._sd = None

    def setup(self):
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
        except ImportError:
            self.status = "vosk/sounddevice not installed - keyboard only"
            return False
        if not os.path.isdir(self.model_dir):
            self.status = f"model not found at {self.model_dir} - keyboard only"
            return False
        try:
            model = Model(self.model_dir)
            grammar = json.dumps(vosk_phrases() + ["[unk]"])
            self._rec = KaldiRecognizer(model, cfg.VOICE_SAMPLE_RATE, grammar)
            self._sd = sd
            self.available = True
            w = cfg.VOICE_WAKE_WORD
            self.status = (f'listening - say "{w} ..." ' if w else
                           "listening")
            return True
        except Exception as e:                      # noqa: BLE001
            self.status = f"voice init failed: {e}"
            return False

    def _callback(self, indata, frames, time_info, status):
        self._q.put(bytes(indata))

    def run(self):
        if not self.setup():
            print(f"[voice] {self.status}")
            return
        print(f"[voice] {self.status} ({len(vosk_phrases())} phrases)")
        try:
            with self._sd.RawInputStream(
                    samplerate=cfg.VOICE_SAMPLE_RATE, blocksize=4000,
                    dtype="int16", channels=1, callback=self._callback):
                while not self._stop_evt.is_set():
                    try:
                        data = self._q.get(timeout=0.3)
                    except queue.Empty:
                        continue
                    if self._rec.AcceptWaveform(data):
                        # FINAL result (after a short pause): most accurate
                        text = json.loads(self._rec.Result()).get("text", "")
                    else:
                        # Partial results can still change ("ana fo..." ->
                        # "follow" / "freeze"), so only "stop" is acted on
                        # early - a wrong stop is harmless, a slow one is not.
                        text = json.loads(
                            self._rec.PartialResult()).get("partial", "")
                        if not text or match_phrase(text) != "STOP":
                            continue
                    cmd = match_phrase(text)
                    if text and text.replace("[unk]", "").strip():
                        self.heard = text.replace("[unk]", "").strip()
                        self.status = f'heard "{self.heard}"' + (
                            "" if cmd else "  (not a command)")
                    if cmd:
                        now = time.time()
                        # one utterance can match on a partial AND the final
                        # result: ignore the same command again within 1 s
                        if not (cmd == self._last_cmd[0] and
                                now - self._last_cmd[1] < 1.0):
                            self.bus.push(cmd, source="voice")
                            self._beep()
                        self._last_cmd = (cmd, now)
                        self._rec.Reset()
        except Exception as e:                      # noqa: BLE001
            self.status = f"voice stopped: {e}"
            print(f"[voice] {self.status}")

    @staticmethod
    def _beep():
        """Short confirmation beep (Windows), never blocks the listener."""
        if not getattr(cfg, "VOICE_BEEP", True) or os.name != "nt":
            return
        try:
            import winsound
            threading.Thread(target=winsound.Beep, args=(1200, 70),
                             daemon=True).start()
        except Exception:                                   # noqa: BLE001
            pass

    def stop(self):
        self._stop_evt.set()
