"""water-log-learner core: features, training, prediction, self-check.

rmaher001/ha-addons#1, rmaher001/ha-automations#712. Pure functions over the
water log's records (one dict per use, as in /config/water_log/YYYY-MM.jsonl)
and Richard's labels (start -> label, newest per start already chosen).

Training data: every non-tiny use with its RULE label (weight 1), except that a
use Richard labelled carries HIS label at HUMAN_WEIGHT. So with no labels the
model reproduces the rules, and his corrections both override them and teach
kinds the rules never had (washer, dishwasher, his own names).
"""
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

FEATURES = ("log_duration", "log_gallons", "median_gpm", "peak_gpm", "gal_per_s",
            "hour_sin", "hour_cos", "thursday")
HUMAN_WEIGHT = 25.0
# A sensor label (learner/sensors.py) is confirmed by the house's own sensors:
# trusted well above a rule label, below Richard's.
SENSOR_WEIGHT = 10.0
THRESHOLD = 0.6
NOT_TRAINED = ("tiny",)


def features(u, tz):
    t = datetime.fromisoformat(u["start"]).astimezone(ZoneInfo(tz))
    hour = t.hour + t.minute / 60.0
    dur = max(float(u["duration_s"]), 1.0)
    gal = max(float(u["gallons"]), 0.0)
    return [math.log1p(dur), math.log1p(gal), float(u["median_gpm"]), float(u["peak_gpm"]),
            gal / dur, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            # Only Thursday matters (the softener's night). A full weekday let a
            # handful of labels teach "washer happens on Tuesdays" instead of
            # what a washer looks like.
            1.0 if t.weekday() == 3 else 0.0]


def trainable(records):
    return [u for u in records if u.get("label") not in NOT_TRAINED]


def _weight(start, human, sensors):
    if start in human:
        return HUMAN_WEIGHT
    return SENSOR_WEIGHT if start in sensors else 1.0


def _dataset(records, human, tz, sensors=None):
    """Each use's target: Richard's label, else the sensors', else the rules'."""
    human = {k: v for k, v in human.items() if v}       # a null label is no label
    sensors = {k: v for k, v in (sensors or {}).items() if v}
    rows = trainable(records)
    X = np.array([features(u, tz) for u in rows], dtype=float)
    y = np.array([human.get(u["start"]) or sensors.get(u["start"]) or u["label"] for u in rows], dtype=object)
    w = np.array([_weight(u["start"], human, sensors) for u in rows])
    return rows, X, y, w


def _fit(X, y, w):
    # min_samples_leaf counts ROWS, not weight: a kind Richard has labelled only a
    # handful of times (the new ones he teaches) must still be able to form a leaf.
    # early_stopping off: above 10k rows "auto" turns it on and holds out a
    # stratified validation split, which raises on any kind with ONE example
    # (Richard's first "washer") and would hide some of his labels from training.
    # l2_regularization and max_leaf_nodes keep it stable. With early stopping off
    # (it crashes on a one-example kind) and no regularisation, 200 rounds
    # diverged on the real history: confident p=1.0 guesses agreeing with the
    # rules on only 84% of uses (2026-11-17). These settings agree on 100%.
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, min_samples_leaf=3,
                                           l2_regularization=1.0, max_leaf_nodes=15,
                                           early_stopping=False, random_state=0)
    model.fit(X, y, sample_weight=w)
    return model


def train(records, human, tz, sensors=None):
    _rows, X, y, w = _dataset(records, human, tz, sensors)
    return _fit(X, y, w)


def predict(model, uses, tz):
    """[(label, probability)] for each use."""
    if not uses:
        return []
    X = np.array([features(u, tz) for u in uses], dtype=float)
    proba = model.predict_proba(X)
    best = proba.argmax(axis=1)
    return [(str(model.classes_[i]), float(proba[k, i])) for k, i in enumerate(best)]


SANITY_MIN = 0.95


def sanity(records, human, preds_rows, preds, sensors=None):
    """Agreement with the rule label on uses Richard has NOT labelled, the rules
    did NOT call unknown, and where the model did NOT pick a kind he taught.

    Spreading his corrections to look-alikes (200 rule "faucet" relabelled
    "washer" moves their twins too) and resolving the rules' unknowns is the
    point of labelling, so neither counts. What is left is divergence on kinds
    he never touched - the 0.1.0 failure: it scored 0.888 here on the real
    history (2026-11-17); a healthy model scores 1.0. The sensors teach too
    (870 rule "unknown" fills are washer): their uses are left out, and so are
    the kinds they teach that the rules never had (washer, dishwasher). Kinds
    the rules DO have (shower, bath) stay checked, or a model calling every
    faucet a shower would pass."""
    sensors = {k: v for k, v in (sensors or {}).items() if v}
    rule_kinds = set([u["label"] for u in records])
    taught = set([v for v in human.values() if v] + [v for v in sensors.values() if v not in rule_kinds])
    pairs = [(u["label"], lab) for u, (lab, _p) in zip(preds_rows, preds)
             if u["start"] not in human and u["start"] not in sensors
             and u["label"] != "unknown" and lab not in taught]
    return sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 1.0


def apply_threshold(preds):
    """A label below THRESHOLD is None: the rule label stands."""
    return [label if p >= THRESHOLD else None for label, p in preds]


def changed_lines(starts, preds, previous, model_at):
    """Append-only lines for guesses that are new or changed since `previous`
    (start -> stored label, None included)."""
    out, seen = [], set()
    for start, label, (_best, p) in zip(starts, apply_threshold(preds), preds):
        if start in seen or (start in previous and previous[start] == label):
            continue
        seen.add(start)
        out.append({"start": start, "label": label, "p": round(p, 3), "model_at": model_at})
    return out


def latest(lines, at_key):
    """start -> label, newest `at_key` per start; malformed lines skipped."""
    best = {}
    for line in lines:
        if (not isinstance(line, dict) or not isinstance(line.get("start"), str)
                or not isinstance(line.get(at_key), str)):
            continue
        label = line.get("label")
        if label is not None and not isinstance(label, str):
            continue
        try:                                   # compare TIMES, not strings ("Z" vs "+00:00")
            at = datetime.fromisoformat(line[at_key].replace("Z", "+00:00"))
        except ValueError:
            continue
        if at.tzinfo is None:                  # a time with no offset is UTC, never naive-vs-aware
            at = at.replace(tzinfo=timezone.utc)
        cur = best.get(line["start"])
        if cur is None or at >= cur[0]:
            best[line["start"]] = (at, label)
    return {s: lab for s, (_a, lab) in best.items()}


def self_check(records, human, tz, folds=5, sensors=None):
    """Cross-validated accuracy on Richard's labels only, or None if there are
    fewer than 2 of them. Each fold holds out some of HIS labelled uses (their
    rule-labelled copies are left out of training too) and scores the model on
    them."""
    human = {k: v for k, v in human.items() if v}
    rows, X, y, w = _dataset(records, human, tz, sensors)
    hidx = np.array([i for i, u in enumerate(rows) if u["start"] in human])
    if len(hidx) < 2:
        return None
    k = min(folds, len(hidx))
    order = np.random.RandomState(0).permutation(hidx)
    chunks = np.array_split(order, k)
    right = 0
    for held in chunks:
        mask = np.ones(len(rows), dtype=bool)
        mask[held] = False
        if len(set(y[mask])) < 2:          # a fold with one class cannot be fitted
            return None
        model = _fit(X[mask], y[mask], w[mask])
        right += int((model.predict(X[held]) == y[held]).sum())
    return right / len(hidx)
