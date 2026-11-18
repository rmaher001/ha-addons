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
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

FEATURES = ("log_duration", "log_gallons", "median_gpm", "peak_gpm", "gal_per_s",
            "hour_sin", "hour_cos", "thursday")
HUMAN_WEIGHT = 25.0
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


def _dataset(records, human, tz):
    rows = trainable(records)
    X = np.array([features(u, tz) for u in rows], dtype=float)
    y = np.array([human.get(u["start"]) or u["label"] for u in rows], dtype=object)
    w = np.array([HUMAN_WEIGHT if u["start"] in human else 1.0 for u in rows])
    return rows, X, y, w


def _fit(X, y, w):
    # min_samples_leaf counts ROWS, not weight: a kind Richard has labelled only a
    # handful of times (the new ones he teaches) must still be able to form a leaf.
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, min_samples_leaf=3, random_state=0)
    model.fit(X, y, sample_weight=w)
    return model


def train(records, human, tz):
    _rows, X, y, w = _dataset(records, human, tz)
    return _fit(X, y, w)


def predict(model, uses, tz):
    """[(label, probability)] for each use."""
    if not uses:
        return []
    X = np.array([features(u, tz) for u in uses], dtype=float)
    proba = model.predict_proba(X)
    best = proba.argmax(axis=1)
    return [(str(model.classes_[i]), float(proba[k, i])) for k, i in enumerate(best)]


def apply_threshold(preds):
    """A label below THRESHOLD is None: the rule label stands."""
    return [label if p >= THRESHOLD else None for label, p in preds]


def changed_lines(starts, preds, previous, model_at):
    """Append-only lines for guesses that are new or changed since `previous`
    (start -> stored label, None included)."""
    out = []
    for start, label, (_best, p) in zip(starts, apply_threshold(preds), preds):
        if start in previous and previous[start] == label:
            continue
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
        cur = best.get(line["start"])
        if cur is None or line[at_key] >= cur[0]:
            best[line["start"]] = (line[at_key], label)
    return {s: lab for s, (_a, lab) in best.items()}


def self_check(records, human, tz, folds=5):
    """Cross-validated accuracy on Richard's labels only, or None if there are
    fewer than 2 of them. Each fold holds out some of HIS labelled uses (their
    rule-labelled copies are left out of training too) and scores the model on
    them."""
    rows, X, y, w = _dataset(records, human, tz)
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
        model = _fit(X[mask], y[mask], w[mask])
        right += int((model.predict(X[held]) == y[held]).sum())
    return right / len(hidx)
