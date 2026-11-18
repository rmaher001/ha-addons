"""water-log-learner: learns Richard's water-use labels (rmaher001/ha-addons#1,
rmaher001/ha-automations#712). Spec: ha-automations
docs/superpowers/specs/2026-11-17-water-log-labelling-learning-design.md, section 3."""
import json
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON = ROOT / "water_log_learner"
sys.path.insert(0, str(ADDON))

from learner import core  # noqa: E402  (a missing module must FAIL, not skip)


def use(start, label, dur, gal, med, peak, source="backfill"):
    return {"start": start, "end": start, "duration_s": dur, "gallons": gal, "median_gpm": med,
            "peak_gpm": peak, "label": label, "rules_version": 2, "source": source}


def history():
    """Flushes, faucets and showers as the rules label them, plus a 'washer'
    pattern the rules call unknown: ~2.8 gal/min for about 2 minutes."""
    out = []
    for d in range(1, 29):
        for h in (7, 12, 19):
            out.append(use(f"2026-10-{d:02d}T{h:02d}:00:00+00:00", "flush", 18, 1.0, 3.5, 3.6))
            out.append(use(f"2026-10-{d:02d}T{h:02d}:10:00+00:00", "faucet", 40, 0.9, 1.4, 1.5))
        out.append(use(f"2026-10-{d:02d}T14:00:00+00:00", "shower", 480, 11.0, 1.4, 4.0))
        out.append(use(f"2026-10-{d:02d}T15:00:00+00:00", "unknown", 120, 5.6, 2.8, 3.0))
    return out


@pytest.mark.requirement("water_log_learner: features are numeric and include time of day and a Thursday flag, not the weekday")
def test_features():
    f = core.features(use("2026-10-05T19:30:00+00:00", "flush", 18, 1.0, 3.5, 3.6), "America/Los_Angeles")
    assert len(f) == len(core.FEATURES) and all(isinstance(x, float) for x in f)
    assert "weekday" not in core.FEATURES and f[core.FEATURES.index("thursday")] == 0.0   # 2026-10-05 is a Monday
    thu = core.features(use("2026-10-08T10:00:00+00:00", "flush", 18, 1.0, 3.5, 3.6), "America/Los_Angeles")
    assert thu[core.FEATURES.index("thursday")] == 1.0


@pytest.mark.requirement("water_log_learner: with no labels from Richard the model reproduces the rules")
def test_trains_on_rules():
    m = core.train(history(), {}, "America/Los_Angeles")
    preds = core.predict(m, [use("2026-11-01T19:00:00+00:00", "?", 17, 0.95, 3.4, 3.5)], "America/Los_Angeles")
    assert preds[0][0] == "flush" and preds[0][1] >= core.THRESHOLD


@pytest.mark.requirement("water_log_learner: a few labels from Richard teach a kind the rules never had")
def test_learns_a_new_kind():
    h = history()
    human = dict([(u["start"], "washer") for u in h if u["label"] == "unknown"][:8])
    m = core.train(h, human, "America/Los_Angeles")
    preds = core.predict(m, [use("2026-11-02T15:00:00+00:00", "?", 125, 5.8, 2.8, 3.0)], "America/Los_Angeles")
    assert preds[0][0] == "washer"


@pytest.mark.requirement("water_log_learner: an unsure guess is stored as None so the rule label stands")
def test_threshold():
    labels = core.apply_threshold([("flush", 0.9), ("faucet", 0.4)])
    assert labels == ["flush", None]


@pytest.mark.requirement("water_log_learner: only new or changed guesses are written, append-only")
def test_changed_lines_only():
    previous = {"A": "flush", "B": None}
    lines = core.changed_lines(["A", "B", "C"], [("flush", 0.9), ("faucet", 0.8), ("shower", 0.3)],
                               previous, "2026-11-18T08:00:00+00:00")
    assert [(l["start"], l["label"]) for l in lines] == [("B", "faucet"), ("C", None)]
    assert all(l["model_at"] == "2026-11-18T08:00:00+00:00" for l in lines)


@pytest.mark.requirement("water_log_learner: the self-check scores only Richard's labels, never the rules'")
def test_self_check():
    h = history()
    human = dict([(u["start"], "washer") for u in h if u["label"] == "unknown"][:10])
    acc = core.self_check(h, human, "America/Los_Angeles")
    assert acc is not None and acc >= 0.8
    assert core.self_check(h, {}, "America/Los_Angeles") is None
    assert core.self_check(h, dict(list(human.items())[:1]), "America/Los_Angeles") is None   # too few to score


@pytest.mark.requirement("water_log_learner: tiny uses are neither trained on nor guessed")
def test_tiny_excluded():
    h = history() + [use("2026-10-03T03:00:00+00:00", "tiny", 5, 0.03, 0.5, 0.5)]
    rows = core.trainable(h)
    assert all(u["label"] != "tiny" for u in rows)


@pytest.mark.requirement("water_log_learner: the App mounts HA config read-write only to write its two files, amd64, no options")
def test_addon_shape():
    cfg = yaml.safe_load((ADDON / "config.yaml").read_text())
    assert cfg["slug"] == "water_log_learner" and cfg["arch"] == ["amd64"]
    assert cfg["map"] == [{"type": "homeassistant_config", "read_only": False}]
    assert "options" not in cfg and "schema" not in cfg
    assert "ports" not in cfg and "host_network" not in cfg and "privileged" not in cfg
    assert cfg["init"] is True, "a bare Python image needs the Supervisor's init for SIGTERM"
    docker = (ADDON / "Dockerfile").read_text()
    assert "scikit-learn==" in docker and "numpy==" in docker, "pinned"


from learner import main as learner_main  # noqa: E402


def _write_month(d, recs):
    with open(d / "2026-10.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


@pytest.mark.requirement("water_log_learner: a pass trains, writes guesses and status, and never touches the month files or Richard's labels")
def test_loop_first_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    (tmp_path / "labels.jsonl").write_text("")
    month_before = (tmp_path / "2026-10.jsonl").read_bytes()
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    guesses = [json.loads(x) for x in (tmp_path / "model_labels.jsonl").read_text().splitlines()]
    assert len(guesses) == len(history())
    status = json.loads((tmp_path / "model_status.json").read_text())
    assert status["labels"] == 0 and status["accuracy"] is None
    assert (tmp_path / "2026-10.jsonl").read_bytes() == month_before
    assert (tmp_path / "labels.jsonl").read_text() == ""
    lrn.tick(now=1793000030)                                   # nothing changed: nothing written
    assert len((tmp_path / "model_labels.jsonl").read_text().splitlines()) == len(history())


@pytest.mark.requirement("water_log_learner: a new use is guessed on the next pass; Richard's label retrains only after a quiet minute")
def test_loop_new_use_and_debounced_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    h = history()
    _write_month(tmp_path, h)
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    n = len((tmp_path / "model_labels.jsonl").read_text().splitlines())
    with open(tmp_path / "2026-10.jsonl", "a") as fh:
        fh.write(json.dumps(use("2026-10-30T19:00:00+00:00", "flush", 18, 1.0, 3.5, 3.6)) + "\n")
    lrn.tick(now=1793000030)
    assert len((tmp_path / "model_labels.jsonl").read_text().splitlines()) == n + 1
    unknowns = [u["start"] for u in h if u["label"] == "unknown"][:10]
    (tmp_path / "labels.jsonl").write_text("".join(
        json.dumps({"start": s, "label": "washer", "at": "2026-11-18T06:00:00+00:00", "by": "richard"}) + "\n"
        for s in unknowns))
    labelled_at = os.path.getmtime(tmp_path / "labels.jsonl")
    lrn.tick(now=labelled_at + 10)                              # too soon: no retrain yet
    assert json.loads((tmp_path / "model_status.json").read_text())["labels"] == 0
    lrn.tick(now=labelled_at + 61)
    status = json.loads((tmp_path / "model_status.json").read_text())
    assert status["labels"] == 10 and status["accuracy"] >= 0.8
    latest = learner_main.core.latest([json.loads(x) for x in (tmp_path / "model_labels.jsonl").read_text().splitlines()], "model_at")
    assert sum(1 for s, lab in latest.items() if lab == "washer") >= 20     # the other washer-shaped uses follow


@pytest.mark.requirement("water_log_learner: the nightly retrain happens once a night")
def test_loop_nightly(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    calls = []
    monkeypatch.setattr(lrn, "retrain", lambda recs: calls.append(1))
    night = datetime_ts("2026-11-18T03:31:00-08:00")
    lrn.tick(now=night)
    lrn.tick(now=night + 30)
    assert calls == [1]


def datetime_ts(iso):
    from datetime import datetime
    return datetime.fromisoformat(iso).timestamp()


import os  # noqa: E402



def big_history():
    """Over 10,000 rows - the size where the library's early stopping would turn on."""
    out = []
    for d in range(1, 29):
        for k in range(400):
            m, sec = divmod(k * 7, 60)
            out.append(use(f"2026-10-{d:02d}T{(k % 24):02d}:{m % 60:02d}:{sec:02d}+00:00",
                           "flush" if k % 2 else "faucet", 18 if k % 2 else 40, 1.0 if k % 2 else 0.9,
                           3.5 if k % 2 else 1.4, 3.6 if k % 2 else 1.5))
    return out


@pytest.mark.requirement("water_log_learner: at real size (>10k rows) a kind with ONE label from Richard trains and is guessed")
def test_singleton_kind_at_real_size():
    h = big_history()
    assert len(h) > 10000
    odd = use("2026-10-15T23:59:00+00:00", "unknown", 120, 5.6, 2.8, 3.0)
    h.append(odd)
    m = core.train(h, {odd["start"]: "washer"}, "America/Los_Angeles")
    assert "washer" in list(m.classes_)
    assert core.self_check(h, {odd["start"]: "washer"}, "America/Los_Angeles") is None   # one label: nothing to score


@pytest.mark.requirement("water_log_learner: a null label from Richard is no label - it neither overrides nor gets his weight")
def test_null_human_label_ignored():
    h = history()
    rows, X, y, w = core._dataset(h, {h[0]["start"]: None}, "America/Los_Angeles")
    assert y[0] == h[0]["label"] and w[0] == 1.0


@pytest.mark.requirement("water_log_learner: a duplicated start is written once per pass")
def test_changed_lines_dedupes():
    lines = core.changed_lines(["A", "A"], [("flush", 0.9), ("flush", 0.9)], {}, "2026-11-18T08:00:00+00:00")
    assert len(lines) == 1


@pytest.mark.requirement("water_log_learner: newest is decided by time, not by string ('Z' vs '+00:00')")
def test_latest_compares_times():
    lines = [{"start": "A", "label": "flush", "at": "2026-11-18T06:00:00.500000+00:00"},
             {"start": "A", "label": "faucet", "at": "2026-11-18T06:00:01Z"}]
    assert core.latest(lines, "at") == {"A": "faucet"}


@pytest.mark.requirement("water_log_learner: the nightly retrain never fires outside 03:30-04:30")
def test_nightly_window_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    lrn = learner_main.Learner()
    lrn.tick(now=datetime_ts("2026-11-17T22:00:00-08:00"))      # first pass trains (no model yet)
    calls = []
    monkeypatch.setattr(lrn, "retrain", lambda recs: calls.append(1))
    for iso in ("2026-11-17T22:30:00-08:00", "2026-11-18T03:29:00-08:00", "2026-11-18T04:31:00-08:00"):
        lrn.tick(now=datetime_ts(iso))
    assert calls == []


@pytest.mark.requirement("water_log_learner: a failing retrain is not retried every 30 s")
def test_failed_retrain_not_hammered(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    (tmp_path / "labels.jsonl").write_text(json.dumps({"start": history()[0]["start"], "label": "x", "at": "2026-11-18T06:00:00+00:00"}) + "\n")
    t0 = os.path.getmtime(tmp_path / "labels.jsonl")
    calls = []

    def boom(recs):
        calls.append(1)
        raise RuntimeError("fit failed")
    monkeypatch.setattr(lrn, "retrain", boom)
    for k in range(3):
        try:
            lrn.tick(now=t0 + 61 + 30 * k)
        except RuntimeError:
            pass
    assert calls == [1]


@pytest.mark.requirement("water_log_learner: deleting labels.jsonl retrains (back to the rules)")
def test_deleted_labels_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    (tmp_path / "labels.jsonl").write_text("")
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    calls = []
    monkeypatch.setattr(lrn, "retrain", lambda recs: calls.append(1))
    (tmp_path / "labels.jsonl").unlink()
    lrn.tick(now=1793000030)
    assert calls == [1]


@pytest.mark.requirement("water_log_learner: a torn last line is repaired before appending; the status file is world-readable")
def test_append_repair_and_status_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    f = tmp_path / "model_labels.jsonl"
    f.write_text('{"start": "A", "lab')
    learner_main.append_lines(str(f), [{"start": "B", "label": "flush", "model_at": "2026-11-18T06:00:00+00:00"}])
    assert json.loads(f.read_text().splitlines()[-1])["start"] == "B"
    learner_main.write_status({"labels": 0})
    assert oct((tmp_path / "model_status.json").stat().st_mode & 0o777) == "0o644"


@pytest.mark.requirement("water_log_learner: a failing FIRST training backs off instead of retrying every pass")
def test_failing_first_training_backs_off(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    lrn = learner_main.Learner()
    calls = []

    def boom(recs):
        calls.append(1)
        raise RuntimeError("fit failed")
    monkeypatch.setattr(lrn, "retrain", boom)
    for k in range(5):
        try:
            lrn.tick(now=1793000000 + 30 * k)
        except RuntimeError:
            pass
    assert calls == [1]
    try:
        lrn.tick(now=1793000000 + learner_main.FIRST_RETRY_S + 1)
    except RuntimeError:
        pass
    assert calls == [1, 1]


@pytest.mark.requirement("water_log_learner: a time without an offset is UTC and never breaks the comparison")
def test_latest_naive_and_aware():
    lines = [{"start": "A", "label": "flush", "at": "2026-11-18T06:00:00"},
             {"start": "A", "label": "faucet", "at": "2026-11-18T06:00:01+00:00"},
             {"start": "B", "label": "washer", "at": "2026-11-18T07:00:00+00:00"},
             {"start": "B", "label": "bath", "at": "2026-11-18T06:00:00"}]
    assert core.latest(lines, "at") == {"A": "faucet", "B": "washer"}


@pytest.mark.requirement("water_log_learner: a stop request mid-write waits for the line to finish, then exits")
def test_stop_during_write_finishes_the_line(tmp_path, monkeypatch):
    f = tmp_path / "model_labels.jsonl"
    learner_main.STOP["now"] = True
    try:
        with pytest.raises(SystemExit):
            learner_main.append_lines(str(f), [{"start": "B", "label": "flush", "model_at": "2026-11-18T06:00:00+00:00"}])
        assert json.loads(f.read_text().splitlines()[-1])["start"] == "B"
    finally:
        learner_main.STOP["now"] = False
