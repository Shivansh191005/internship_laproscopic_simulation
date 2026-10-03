"""
Check that every voice command is really recognised - without speaking.

    python voice_test.py              # computer voice says every command
    python voice_test.py --mic        # you speak; prints what was heard

The first mode makes a computer voice (Windows SAPI via pyttsx3, or
espeak-ng on Linux) say each phrase into a WAV file, runs it through the
same Vosk recogniser + grammar + matching as the live program and prints a
PASS/FAIL table. Real voices are clearer than this robotic one, so if these
pass, the live microphone usually works even better.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg                                         # noqa: E402
from commands import GRAMMAR, STOP_WORD, match_phrase, vosk_phrases  # noqa


# how the computer voice must SPELL a word to pronounce it right
# (Ana = "AH-nuh"; written "ana" the voice says "ay-na")
SAY_AS = {"ana": "ahna"}


def _tts(text, path):
    """Say `text` into a 16 kHz mono WAV. Returns True on success."""
    text = " ".join(SAY_AS.get(w, w) for w in text.split())
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", 150)
        eng.save_to_file(text, path)
        eng.runAndWait()
        if os.path.getsize(path) > 1000:
            return True
    except Exception:                                       # noqa: BLE001
        pass
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe:
        subprocess.run([exe, "-s", "140", "-w", path, text],
                       capture_output=True)
        return os.path.isfile(path) and os.path.getsize(path) > 1000
    return False


def _pcm16k(path):
    """Read a WAV and return 16 kHz mono int16 bytes."""
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        a = np.frombuffer(w.readframes(w.getnframes()),
                          {1: np.uint8, 2: np.int16, 4: np.int32}[sw])
    a = a.astype(np.float32)
    if sw == 1:
        a = (a - 128) * 256
    elif sw == 4:
        a = a / 65536
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != cfg.VOICE_SAMPLE_RATE:
        n = int(len(a) * cfg.VOICE_SAMPLE_RATE / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    pad = np.zeros(int(0.4 * cfg.VOICE_SAMPLE_RATE))      # silence around
    return np.concatenate([pad, a, pad]).clip(-32768, 32767).astype(np.int16)\
        .tobytes()


def recognise(rec, pcm):
    """Same rule as voice.py: act on FINAL results (more accurate); only
    "stop" is taken from a partial result, so it reacts instantly."""
    rec.Reset()
    heard, cmd = "", None
    for i in range(0, len(pcm), 8000):
        if rec.AcceptWaveform(pcm[i:i + 8000]):
            t = json.loads(rec.Result()).get("text", "")
            if t:
                heard = t
                cmd = cmd or match_phrase(t)
        else:
            t = json.loads(rec.PartialResult()).get("partial", "")
            if t and match_phrase(t) == "STOP":
                cmd = cmd or "STOP"
    t = json.loads(rec.FinalResult()).get("text", "")
    if t:
        heard = t
        cmd = cmd or match_phrase(t)
    return heard, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="listen to you")
    args = ap.parse_args()
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
    except ImportError:
        print("vosk not installed:  python -m pip install vosk sounddevice")
        return 1
    if not os.path.isdir(cfg.VOSK_MODEL_DIR):
        print(f"model missing: {cfg.VOSK_MODEL_DIR}")
        return 1
    model = Model(cfg.VOSK_MODEL_DIR)
    rec = KaldiRecognizer(model, cfg.VOICE_SAMPLE_RATE,
                          json.dumps(vosk_phrases() + ["[unk]"]))

    if args.mic:
        import sounddevice as sd
        print("Speak commands (Ctrl+C to quit)...")
        with sd.RawInputStream(samplerate=cfg.VOICE_SAMPLE_RATE,
                               blocksize=8000, dtype="int16",
                               channels=1) as s:
            while True:
                data, _ = s.read(8000)
                if rec.AcceptWaveform(bytes(data)):
                    t = json.loads(rec.Result()).get("text", "")
                    if t:
                        print(f"heard: {t!r:30s} -> {match_phrase(t)}")

    w = cfg.VOICE_WAKE_WORD
    cases = [(f"{w} {p}".strip(), c) for p, c in GRAMMAR.items()
             if p not in ("down", "hold", "home")]   # aliases of lower /
    #                                                  freeze / reset
    cases += [(STOP_WORD, "STOP"),
              ("hello doctor", None), ("the robot is fine", None),
              ("zoom in please", None),
              ("please pass the scissors", None)]
    tmp = tempfile.mkdtemp()
    ok = 0
    print(f"{'said':28s} {'heard':28s} {'command':14s} result")
    for i, (text, want) in enumerate(cases):
        p = os.path.join(tmp, f"{i}.wav")
        if not _tts(text, p):
            print("no computer voice available (pip install pyttsx3)")
            return 1
        heard, got = recognise(rec, _pcm16k(p))
        good = got == want
        ok += good
        print(f"{text:28s} {heard:28s} {str(got):14s} "
              f"{'PASS' if good else 'FAIL'}")
    print(f"\n{ok}/{len(cases)} correct "
          f"(last {4} lines check that normal talk is ignored)")
    return 0 if ok == len(cases) else 2


if __name__ == "__main__":
    sys.exit(main())
