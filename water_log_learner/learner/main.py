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
import tempfile
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from learner import core

LOG_DIR = os.environ.get("WATER_LOG_DIR", "/homeassistant/water_log")
TZ = os.environ.get("WATER_LOG_TZ", "America/Los_Angeles")
POLL_S = 30
DEBOUNCE_S = 60
NIGHTLY = (3, 30)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
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
    fd, tmp = tempfile.mkstemp(dir=LOG_DIR, prefix=".model_status.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(status, fh, sort_keys=True)
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

    def guessed(self):
        return core.latest(read_jsonl(os.path.join(LOG_DIR, "model_labels.jsonl")), "model_at")

    def retrain(self, recs):
        human = core.latest(read_jsonl(os.path.join(LOG_DIR, "labels.jsonl")), "at")
        self.model = core.train(recs, human, TZ)
        acc = core.self_check(recs, human, TZ)
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
        changed = labels_m != self.labels_seen and (labels_m is None or now - labels_m >= DEBOUNCE_S)
        recs = records()
        if not recs:
            return
        if self.model is None or changed or night:
            self.retrain(recs)
            self.labels_seen = labels_m
            if night:
                self.last_night = local.date()
        else:
            self.label_new(recs)


def main():
    learner = Learner()
    while True:
        try:
            learner.tick()
        except Exception as err:          # one bad pass must not end the App
            print(f"{now_iso()} pass failed: {err!r}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
