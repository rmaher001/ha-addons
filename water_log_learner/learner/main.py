"""water-log-learner main loop (rmaher001/ha-addons#1).

Every POLL_S seconds:
  * retrain when Richard's labels changed (after DEBOUNCE_S of quiet), at
    NIGHTLY (local time) once a day, and on start; each retrain re-scores
    itself on his labels and relabels all history;
  * otherwise label uses that appeared since the last pass;
  * read the house's sensors (learner/sensors.py) for all history on start and
    nightly, and for the last 48 hours every 10 minutes, appending what they
    confirm to sensor_labels.jsonl and the bathroom they place a shower, bath
    or flush in to room_labels.jsonl. New sensor labels retrain like
    Richard's; rooms never train.
It appends ONLY new or changed lines to model_labels.jsonl, sensor_labels.jsonl
and room_labels.jsonl and rewrites model_status.json. It never writes the month
files or labels.jsonl.
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

from learner import core, sensors

STOP = {"now": False}

LOG_DIR = os.environ.get("WATER_LOG_DIR", "/homeassistant/water_log")
TZ = os.environ.get("WATER_LOG_TZ", "America/Los_Angeles")
POLL_S = 30
DEBOUNCE_S = 60
FIRST_RETRY_S = 600          # a failed FIRST training is retried after 10 min, not every pass
WRITING = {"now": False}     # a stop request waits only while a file is being written
NIGHTLY = (3, 30)
# Where the house's sensor history is kept, read only. A constant in git: an
# add-on here has no options (README, "Update"). None = no sensor labels.
VM_URL = "http://victoriametrics.d.lan.ram6.com:8428"
EVIDENCE = sensors.evidence  # (vm_url, start, end) -> sensors.Evidence; raises when it cannot read
SENSOR_EVERY_S = 600
SENSOR_RECENT_S = 48 * 3600


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
        self.last_status = None
        self.sensors_next = 0.0      # when the next 48-hour sensor read is due
        self.sensors_full = True     # read all history: on start, then nightly

    def sensor_labels(self):
        return core.latest(read_jsonl(os.path.join(LOG_DIR, "sensor_labels.jsonl")), "at")

    def room_labels(self):
        return core.latest(read_jsonl(os.path.join(LOG_DIR, "room_labels.jsonl")), "at", key="room")

    def sensor_pass(self, recs, now, full):
        """Append what the sensors confirm; False when it could not. Any
        failure writes nothing: missing evidence is never taken as "it was not
        a shower"."""
        if VM_URL is None:
            return True
        try:
            rows = core.trainable(recs)
            if not full:
                rows = [u for u in rows if sensors.epoch(u["start"]) >= now - SENSOR_RECENT_S]
            if not rows:
                return True
            times = [sensors.epoch(u["start"]) for u in rows]
            ev = EVIDENCE(VM_URL, min(times), max(times))
            starts, at = [u["start"] for u in rows], now_iso()
            # Both worked out before either is written.
            lines = sensors.changed_lines(starts, sensors.label_all(rows, ev), self.sensor_labels(), at)
            rooms = []
            empty = ", ".join(getattr(ev, "empty", ()))
            if not getattr(ev, "rooms_ok", True):
                print(f"{at} rooms skipped: no points for {empty} over the full read", flush=True)
            else:
                placed = starts
                if not getattr(ev, "sinks_ok", True):
                    # Faucet and unknown rooms stay as they are: missing data is never
                    # "nobody at the sink" (an unknown's room needs the kitchen ruled out).
                    print(f"{at} faucet and unknown rooms skipped: no points for {empty} over the full read", flush=True)
                    placed = [u["start"] for u in rows if u.get("label") not in ("faucet", "unknown")]
                rooms = sensors.changed_lines(placed, sensors.room_all(rows, ev), self.room_labels(), at, key="room")
            append_lines(os.path.join(LOG_DIR, "sensor_labels.jsonl"), lines)
            append_lines(os.path.join(LOG_DIR, "room_labels.jsonl"), rooms)
        except Exception as err:
            print(f"{now_iso()} sensor pass failed: {err!r}", flush=True)
            return False
        if lines or rooms:
            print(f"{now_iso()} sensors: {len(lines)} labels, {len(rooms)} rooms written "
                  f"({'all history' if full else 'last 48 h'})", flush=True)
        return True

    def guessed(self):
        return core.latest(read_jsonl(os.path.join(LOG_DIR, "model_labels.jsonl")), "model_at")

    def retrain(self, recs):
        human = core.latest(read_jsonl(os.path.join(LOG_DIR, "labels.jsonl")), "at")
        sens = dict((k, v) for k, v in self.sensor_labels().items() if v)
        model = core.train(recs, human, TZ, sens)
        try:
            acc = core.self_check(recs, human, TZ, sensors=sens)
        except Exception as err:               # the score is a report, never a blocker
            print(f"{now_iso()} self-check failed: {err!r}", flush=True)
            acc = None
        previous_model, self.model = self.model, model
        rows = core.trainable(recs)
        at = now_iso()
        preds = core.predict(model, rows, TZ)
        agree = core.sanity(recs, {k: v for k, v in human.items() if v}, rows, preds, sens)
        if agree < core.SANITY_MIN:
            # A broken model writes NOTHING: the previous guesses (or the rules) stand.
            self.model = previous_model
            # Keep the LAST GOOD run's figures: the guesses in use are still its.
            status = self.last_status or {}
            write_status({**status, "rejected": True, "rejected_at": at, "agree_with_rules": round(agree, 4)})
            print(f"{at} model REJECTED: agrees with the rules on {agree:.1%} of unlabelled uses", flush=True)
            return
        lines = core.changed_lines([u["start"] for u in rows], preds, self.guessed(), at)
        append_lines(os.path.join(LOG_DIR, "model_labels.jsonl"), lines)
        self.last_status = {"trained_at": at, "labels": len(human), "accuracy": acc, "uses": len(rows),
                            "agree_with_rules": round(agree, 4), "sensor_labels": len(sens)}
        write_status(self.last_status)
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
        local = datetime.fromtimestamp(now, ZoneInfo(TZ))
        # Only inside the hour after NIGHTLY: "any time after 3:30 on a day it has
        # not run" would fire at 10 PM on the App's first day.
        minutes = local.hour * 60 + local.minute
        start = NIGHTLY[0] * 60 + NIGHTLY[1]
        night = start <= minutes < start + 60 and self.last_night != local.date()
        recs = records()
        if not recs:
            return
        # Sensors first: on the first pass the training then already sees what
        # they found; later, new sensor labels retrain after a quiet minute.
        if night:
            self.sensors_full = True
        if night or now >= self.sensors_next:
            full, self.sensors_full = self.sensors_full, False
            self.sensors_next = now + SENSOR_EVERY_S
            if not self.sensor_pass(recs, now, full) and full:
                self.sensors_full = True     # the whole history again at the next interval
        # Richard's labels and the sensors' both retrain; the newer file counts.
        stamps = [m for m in (mtime(os.path.join(LOG_DIR, "labels.jsonl")),
                              mtime(os.path.join(LOG_DIR, "sensor_labels.jsonl"))) if m is not None]
        labels_m = max(stamps) if stamps else None
        # Quiet for a minute - or an mtime in the future (clock skew), which
        # would otherwise block retraining for good.
        quiet = labels_m is None or now - labels_m >= DEBOUNCE_S or labels_m - now > DEBOUNCE_S
        changed = labels_m != self.labels_seen and quiet
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
    print(f"{now_iso()} started; sensor labels from {VM_URL or 'nowhere (no vm_url)'}", flush=True)
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
