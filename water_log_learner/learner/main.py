"""water-log-learner main loop (rmaher001/ha-addons#1).

Every POLL_S seconds:
  * retrain when Richard's labels changed (after DEBOUNCE_S of quiet), at
    NIGHTLY (local time) once a day, and on start; each retrain re-scores
    itself on his labels and relabels all history;
  * otherwise label uses that appeared since the last pass.
It appends ONLY new or changed guesses to model_labels.jsonl and rewrites
model_status.json. It never writes the month files or labels.jsonl.
"""
import glob
import json
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from learner import core

STOP = {"now": False}

LOG_DIR = os.environ.get("WATER_LOG_DIR", "/homeassistant/water_log")
TZ = os.environ.get("WATER_LOG_TZ", "America/Los_Angeles")
POLL_S = 30
DEBOUNCE_S = 60
FIRST_RETRY_S = 600          # a failed FIRST training is retried after 10 min, not every pass
WRITING = {"now": False}     # a stop request waits only while a file is being written
NIGHTLY = (3, 30)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    # errors="replace": a torn write that cut a multi-byte character must not
    # fail the whole pass.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def records():
    out = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "[0-9][0-9][0-9][0-9]-[0-9][0-9].jsonl"))):
        for r in read_jsonl(path):
            if isinstance(r.get("start"), str) and "label" in r and "duration_s" in r:
                out.append(r)
    return out


def append_lines(path, lines):
    if not lines:
        return
    WRITING["now"] = True
    try:
        _append_lines(path, lines)
    finally:
        WRITING["now"] = False
    if STOP["now"]:
        sys.exit(0)


def _append_lines(path, lines):
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            broken = fh.read(1) != b"\n"
        if broken:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
    with open(path, "a", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")


def write_status(status):
    WRITING["now"] = True
    try:
        _write_status(status)
    finally:
        WRITING["now"] = False
    if STOP["now"]:
        sys.exit(0)


def _write_status(status):
    fd, tmp = tempfile.mkstemp(dir=LOG_DIR, prefix=".model_status.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(status, fh, sort_keys=True)
    os.chmod(tmp, 0o644)
    os.replace(tmp, os.path.join(LOG_DIR, "model_status.json"))


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


class Learner:
    def __init__(self):
        self.model = None
        self.labels_seen = None
        self.last_night = None
        self.done = set()
        self.retry_after = 0.0

    def guessed(self):
        return core.latest(read_jsonl(os.path.join(LOG_DIR, "model_labels.jsonl")), "model_at")

    def retrain(self, recs):
        human = core.latest(read_jsonl(os.path.join(LOG_DIR, "labels.jsonl")), "at")
        model = core.train(recs, human, TZ)
        try:
            acc = core.self_check(recs, human, TZ)
        except Exception as err:               # the score is a report, never a blocker
            print(f"{now_iso()} self-check failed: {err!r}", flush=True)
            acc = None
        self.model = model
        rows = core.trainable(recs)
        at = now_iso()
        lines = core.changed_lines([u["start"] for u in rows], core.predict(self.model, rows, TZ),
                                   self.guessed(), at)
        append_lines(os.path.join(LOG_DIR, "model_labels.jsonl"), lines)
        write_status({"trained_at": at, "labels": len(human), "accuracy": acc, "uses": len(rows)})
        self.done = {u["start"] for u in rows}
        print(f"{at} retrained on {len(rows)} uses, {len(human)} labels, accuracy {acc}, {len(lines)} guesses written", flush=True)

    def label_new(self, recs):
        fresh = [u for u in core.trainable(recs) if u["start"] not in self.done]
        if not fresh or self.model is None:
            return
        lines = core.changed_lines([u["start"] for u in fresh], core.predict(self.model, fresh, TZ),
                                   self.guessed(), now_iso())
        append_lines(os.path.join(LOG_DIR, "model_labels.jsonl"), lines)
        self.done.update(u["start"] for u in fresh)

    def tick(self, now=None):
        now = now or time.time()
        labels_m = mtime(os.path.join(LOG_DIR, "labels.jsonl"))
        local = datetime.fromtimestamp(now, ZoneInfo(TZ))
        # Only inside the hour after NIGHTLY: "any time after 3:30 on a day it has
        # not run" would fire at 10 PM on the App's first day.
        minutes = local.hour * 60 + local.minute
        start = NIGHTLY[0] * 60 + NIGHTLY[1]
        night = start <= minutes < start + 60 and self.last_night != local.date()
        # Quiet for a minute - or an mtime in the future (clock skew), which
        # would otherwise block retraining for good.
        quiet = labels_m is None or now - labels_m >= DEBOUNCE_S or labels_m - now > DEBOUNCE_S
        changed = labels_m != self.labels_seen and quiet
        recs = records()
        if not recs:
            return
        first = self.model is None and now >= self.retry_after
        if first or changed or night:
            # Count the attempt BEFORE running it: a retrain that fails must not
            # be retried at full cost every 30 s (the next label or night retries).
            self.labels_seen = labels_m
            if night:
                self.last_night = local.date()
            if self.model is None:
                self.retry_after = now + FIRST_RETRY_S
            self.retrain(recs)
        else:
            self.label_new(recs)


def _stop(_signum, _frame):
    # Exit at once - a retrain can take a while and holds nothing on disk -
    # unless a file is being written: then append_lines/write_status exit
    # right after the write completes, so no line is ever cut.
    STOP["now"] = True
    if not WRITING["now"]:
        sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    learner = Learner()
    while not STOP["now"]:
        try:
            learner.tick()
        except Exception as err:          # one bad pass must not end the App
            print(f"{now_iso()} pass failed: {err!r}", flush=True)
        for _ in range(POLL_S):
            if STOP["now"]:
                break
            time.sleep(1)
    print(f"{now_iso()} stopped", flush=True)


if __name__ == "__main__":
    main()
