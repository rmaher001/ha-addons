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


@pytest.fixture(autouse=True)
def _no_sensor_reads(monkeypatch):
    """No test reads the real VictoriaMetrics: sensor tests install a stub."""
    monkeypatch.setattr(learner_main, "VM_URL", None)


@pytest.mark.requirement("water_log_learner: the sensor history address is a DNS name fixed in git, not an option")
def test_vm_url_is_a_constant():
    import importlib
    fresh = importlib.reload(learner_main)
    assert fresh.VM_URL == "http://victoriametrics.d.lan.ram6.com:8428"


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


@pytest.mark.requirement("water_log_learner: a model that disagrees with the rules on unlabelled uses writes NOTHING and is marked rejected")
def test_sanity_gate_rejects_a_broken_model(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    lrn = learner_main.Learner()
    monkeypatch.setattr(learner_main.core, "predict", lambda model, rows, tz: [("faucet", 1.0) for _ in rows])
    lrn.tick(now=1793000000)
    assert not (tmp_path / "model_labels.jsonl").exists()
    status = json.loads((tmp_path / "model_status.json").read_text())
    assert status["rejected"] is True and status["agree_with_rules"] < 0.95
    assert lrn.model is None


@pytest.mark.requirement("water_log_learner: sanity ignores uses Richard labelled and the rules' unknowns (they are SUPPOSED to change)")
def test_sanity_ignores_labelled_and_unknown():
    rows = [use("A", "faucet", 10, 1, 1, 1), use("B", "faucet", 10, 1, 1, 1), use("C", "unknown", 10, 1, 1, 1)]
    assert core.sanity(rows, {"A": "washer"}, rows, [("washer", 1.0), ("faucet", 1.0), ("washer", 1.0)]) == 1.0



@pytest.mark.requirement("water_log_learner: bulk-teaching a new kind out of a rule kind is ACCEPTED, not rejected by the gate")
def test_gate_accepts_bulk_teaching():
    rows = []
    for k in range(1000):
        washer = k % 10 < 3                              # 300 of the rule "faucets" are really a washer
        day, sec = divmod(k, 40)
        rows.append(use(f"2026-10-{1 + day % 28:02d}T{8 + sec % 12:02d}:{sec:02d}:{k % 60:02d}+00:00", "faucet",
                        120 if washer else 40, 5.6 if washer else 0.9, 2.8 if washer else 1.4, 3.0 if washer else 1.5))
    rows += history()
    human = dict([(u["start"], "washer") for u in rows if u["duration_s"] == 120][:30])
    m = core.train(rows, human, "America/Los_Angeles")
    tr = core.trainable(rows)
    preds = core.predict(m, tr, "America/Los_Angeles")
    moved = sum(1 for u, (lab, _p) in zip(tr, preds) if u["duration_s"] == 120 and u["start"] not in human and lab == "washer")
    assert moved > 200, "the look-alikes follow his labels"
    assert core.sanity(rows, human, tr, preds) >= core.SANITY_MIN


@pytest.mark.requirement("water_log_learner: a rejected run keeps the last good run's figures and says when it was rejected")
def test_rejected_status_keeps_last_good(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    good = json.loads((tmp_path / "model_status.json").read_text())
    monkeypatch.setattr(learner_main.core, "predict", lambda model, rows, tz: [("faucet", 1.0) for _ in rows])
    lrn.retrain(learner_main.records())
    status = json.loads((tmp_path / "model_status.json").read_text())
    assert status["rejected"] is True and status["trained_at"] == good["trained_at"] and "rejected_at" in status



# ---- sensor labels (ha-automations#712, "Sensor labels") ---------------------

@pytest.mark.requirement("water_log_learner: sensor labels teach the model a kind, like Richard's do")
def test_sensor_labels_teach():
    h = history()
    sens = dict([(u["start"], "washer") for u in h if u["label"] == "unknown"][:8])
    m = core.train(h, {}, "America/Los_Angeles", sens)
    preds = core.predict(m, [use("2026-11-02T15:00:00+00:00", "?", 125, 5.8, 2.8, 3.0)], "America/Los_Angeles")
    assert preds[0][0] == "washer"


@pytest.mark.requirement("water_log_learner: Richard's label beats the sensors' on the same use, and weighs more")
def test_human_beats_sensors():
    h = history()
    s = h[0]["start"]
    _rows, _X, y, w = core._dataset(h, {s: "faucet"}, "America/Los_Angeles", {s: "washer", h[1]["start"]: "washer"})
    assert y[0] == "faucet" and w[0] == core.HUMAN_WEIGHT
    assert y[1] == "washer" and w[1] == core.SENSOR_WEIGHT and 1.0 < core.SENSOR_WEIGHT < core.HUMAN_WEIGHT


@pytest.mark.requirement("water_log_learner: the sanity gate leaves out sensor-labelled uses and the kinds the sensors teach")
def test_sanity_ignores_sensor_kinds():
    rows = [use("a", "faucet", 40, 0.9, 1.4, 1.5), use("b", "faucet", 40, 0.9, 1.4, 1.5), use("c", "flush", 18, 1.0, 3.5, 3.6)]
    preds = [("dishwasher", 0.9), ("dishwasher", 0.9), ("flush", 0.9)]
    assert core.sanity(rows, {}, rows, preds, {"a": "dishwasher"}) == 1.0
    assert core.sanity(rows, {}, rows, preds) < 1.0


def _sensor_stub(monkeypatch, calls, fail=False):
    def fake_evidence(url, a, b):
        calls.append((url, a, b))
        if fail:
            raise OSError("VictoriaMetrics unreachable")
        return "EV"
    monkeypatch.setattr(learner_main, "VM_URL", "http://vm.test:8428")
    monkeypatch.setattr(learner_main, "EVIDENCE", fake_evidence)


@pytest.mark.requirement("water_log_learner: the App writes sensor labels for all history on start, trains on them, and counts them in its status")
def test_sensor_pass_on_start(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    h = history()
    _write_month(tmp_path, h)
    calls = []
    _sensor_stub(monkeypatch, calls)
    washers = [u["start"] for u in h if u["label"] == "unknown"][:10]
    monkeypatch.setattr(learner_main.sensors, "label_all",
                        lambda uses, ev: dict((u["start"], ("washer", "washer cycle")) for u in uses if u["start"] in washers))
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    lines = [json.loads(x) for x in (tmp_path / "sensor_labels.jsonl").read_text().splitlines()]
    assert sorted(x["start"] for x in lines) == sorted(washers)
    assert all(x["by"] == "sensors" and x["label"] == "washer" and x["why"] == "washer cycle" for x in lines)
    assert len(calls) == 1 and calls[0][0] == "http://vm.test:8428"
    status = json.loads((tmp_path / "model_status.json").read_text())
    assert status["sensor_labels"] == 10 and status["labels"] == 0
    latest = learner_main.core.latest([json.loads(x) for x in (tmp_path / "model_labels.jsonl").read_text().splitlines()], "model_at")
    assert sum(1 for lab in latest.values() if lab == "washer") >= 20      # the look-alikes follow
    lrn.tick(now=1793000030)                                   # within 10 min: no second read
    assert len(calls) == 1


@pytest.mark.requirement("water_log_learner: every 10 minutes only the last 48 hours are re-read")
def test_sensor_pass_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    now = datetime_ts("2026-10-29T12:00:00+00:00")
    _write_month(tmp_path, history())
    calls = []
    _sensor_stub(monkeypatch, calls)
    monkeypatch.setattr(learner_main.sensors, "label_all", lambda uses, ev: {})
    lrn = learner_main.Learner()
    lrn.tick(now=now)
    lrn.tick(now=now + learner_main.SENSOR_EVERY_S + 1)
    assert len(calls) == 2
    assert calls[0][1] == datetime_ts("2026-10-01T07:00:00+00:00")             # all history
    assert calls[1][1] >= now - learner_main.SENSOR_RECENT_S                     # the last 48 h


@pytest.mark.requirement("water_log_learner: VictoriaMetrics unreachable writes no sensor labels and the model still trains")
def test_sensor_pass_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    calls = []
    _sensor_stub(monkeypatch, calls, fail=True)
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    assert not (tmp_path / "sensor_labels.jsonl").exists()
    assert json.loads((tmp_path / "model_status.json").read_text())["sensor_labels"] == 0
    lrn.tick(now=1793000000 + learner_main.SENSOR_EVERY_S + 1)              # retried at the next interval
    assert len(calls) == 2


@pytest.mark.requirement("water_log_learner: new sensor labels retrain the model after a quiet minute, like Richard's")
def test_sensor_labels_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    h = history()
    _write_month(tmp_path, h)
    calls = []
    _sensor_stub(monkeypatch, calls)
    found = {}
    monkeypatch.setattr(learner_main.sensors, "label_all", lambda uses, ev: dict(found))
    now = datetime_ts("2026-10-29T00:00:00+00:00")
    lrn = learner_main.Learner()
    lrn.tick(now=now)
    for u in [u for u in h if u["label"] == "unknown"][-2:]:  # Oct 27 and 28: inside the last 48 h
        found[u["start"]] = ("washer", "washer cycle")
    t = now + learner_main.SENSOR_EVERY_S + 1
    lrn.tick(now=t)                                            # writes the labels
    assert len((tmp_path / "sensor_labels.jsonl").read_text().splitlines()) == 2
    written = os.path.getmtime(tmp_path / "sensor_labels.jsonl")
    lrn.tick(now=max(t, written) + learner_main.DEBOUNCE_S + 1)
    assert json.loads((tmp_path / "model_status.json").read_text())["sensor_labels"] == 2


@pytest.mark.requirement("water_log_sensors: no dishwasher-plug points at all means the store did not answer: the read fails")
def test_evidence_needs_the_plug(monkeypatch):
    from learner import sensors
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: [] if "dishwasher" in match else [(1.0, 1.0)])
    with pytest.raises(RuntimeError):
        sensors.evidence("http://vm.test", 0, 10)


@pytest.mark.requirement("water_log_learner: a failed all-history sensor read is retried at the next interval, not left until the night")
def test_failed_full_read_is_retried_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    now = datetime_ts("2026-10-29T12:00:00+00:00")
    _write_month(tmp_path, history())
    calls = []
    _sensor_stub(monkeypatch, calls, fail=True)
    lrn = learner_main.Learner()
    lrn.tick(now=now)
    lrn.tick(now=now + learner_main.SENSOR_EVERY_S + 1)
    assert calls[1][1] == datetime_ts("2026-10-01T07:00:00+00:00")             # all history again


@pytest.mark.requirement("water_log_learner: a sensor pass that fails while labelling (not only while reading) writes nothing and the pass still trains")
def test_sensor_labelling_error_is_contained(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    calls = []
    _sensor_stub(monkeypatch, calls)
    def boom(uses, ev):
        raise KeyError("median_gpm")
    monkeypatch.setattr(learner_main.sensors, "label_all", boom)
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    assert not (tmp_path / "sensor_labels.jsonl").exists()
    assert (tmp_path / "model_status.json").exists()


@pytest.mark.requirement("water_log_learner: the sanity gate still checks rule kinds the sensors teach (shower, bath): only kinds new to the rules are exempt")
def test_sanity_checks_rule_kinds_sensors_teach():
    rows = [use(f"f{i}", "faucet", 40, 0.9, 1.4, 1.5) for i in range(10)] + [use("s", "shower", 480, 11.0, 1.4, 4.0)]
    broken = [("shower", 0.9)] * 11                           # calls every faucet a shower
    assert core.sanity(rows, {}, rows, broken, {"s": "shower"}) < core.SANITY_MIN
    new_kind = [("dishwasher", 0.9)] * 10 + [("shower", 0.9)]
    assert core.sanity(rows, {}, rows, new_kind, {"s": "dishwasher"}) == 1.0



@pytest.mark.requirement("water_log_learner: the nightly pass re-reads ALL history from the sensors")
def test_nightly_full_sensor_read(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    calls = []
    _sensor_stub(monkeypatch, calls)
    monkeypatch.setattr(learner_main.sensors, "label_all", lambda uses, ev: {})
    lrn = learner_main.Learner()
    lrn.tick(now=datetime_ts("2026-10-29T10:00:00-07:00"))
    lrn.tick(now=datetime_ts("2026-10-30T03:31:00-07:00"))
    assert len(calls) == 2 and calls[1][1] == datetime_ts("2026-10-01T07:00:00+00:00")


@pytest.mark.requirement("water_log_learner: with no store address the App never tries to read sensors")
def test_no_vm_url_no_read(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    _write_month(tmp_path, history())
    calls = []
    monkeypatch.setattr(learner_main, "EVIDENCE", lambda *a: calls.append(a))
    learner_main.Learner().tick(now=1793000000)
    assert calls == [] and not (tmp_path / "sensor_labels.jsonl").exists()


@pytest.mark.requirement("water_log_learner: a failed read keeps every existing sensor label (an outage is never a withdrawal)")
def test_failed_read_keeps_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    h = history()
    _write_month(tmp_path, h)
    before = json.dumps({"start": h[-1]["start"], "label": "washer", "why": "washer cycle", "by": "sensors",
                         "at": "2026-10-28T00:00:00+00:00"}) + "\n"
    (tmp_path / "sensor_labels.jsonl").write_text(before)
    calls = []
    _sensor_stub(monkeypatch, calls, fail=True)
    learner_main.Learner().tick(now=1793000000)
    assert (tmp_path / "sensor_labels.jsonl").read_text() == before


@pytest.mark.requirement("water_log_learner: retrain passes the sensor labels to the self-check and the sanity gate, and counts only live ones")
def test_retrain_wires_sensor_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    h = history()
    _write_month(tmp_path, h)
    a, b = h[0]["start"], h[1]["start"]
    (tmp_path / "sensor_labels.jsonl").write_text(
        json.dumps({"start": a, "label": "washer", "by": "sensors", "at": "2026-10-28T00:00:00+00:00"}) + "\n" +
        json.dumps({"start": b, "label": None, "by": "sensors", "at": "2026-10-28T00:00:00+00:00"}) + "\n")
    seen = {}
    real_sanity, real_check = learner_main.core.sanity, learner_main.core.self_check
    monkeypatch.setattr(learner_main.core, "sanity", lambda *args: seen.update(sanity=args[4]) or real_sanity(*args))
    monkeypatch.setattr(learner_main.core, "self_check",
                        lambda *args, **kw: seen.update(check=kw.get("sensors")) or real_check(*args, **kw))
    learner_main.Learner().retrain(learner_main.records())
    assert seen["sanity"] == {a: "washer"} and seen["check"] == {a: "washer"}
    assert json.loads((tmp_path / "model_status.json").read_text())["sensor_labels"] == 1


@pytest.mark.requirement("water_log_learner: the sanity gate leaves out a sensor-labelled use even when its kind is one the rules have")
def test_sanity_skips_sensor_rows():
    rows = [use("a", "faucet", 40, 0.9, 1.4, 1.5), use("b", "flush", 18, 1.0, 3.5, 3.6),
            use("c", "shower", 480, 11.0, 1.4, 4.0)]          # shower IS a rule kind here
    preds = [("shower", 0.9), ("flush", 0.9), ("shower", 0.9)]
    assert core.sanity(rows, {}, rows, preds, {"a": "shower"}) == 1.0


@pytest.mark.requirement("water_log_learner: a failed write of sensor labels keeps the all-history read due and the pass still trains")
def test_sensor_write_failure_contained(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_main, "LOG_DIR", str(tmp_path))
    h = history()
    _write_month(tmp_path, h)
    calls = []
    _sensor_stub(monkeypatch, calls)
    monkeypatch.setattr(learner_main.sensors, "label_all", lambda uses, ev: {h[-1]["start"]: ("washer", "washer cycle")})
    real = learner_main.append_lines

    def append(path, lines):
        if path.endswith("sensor_labels.jsonl"):
            raise OSError("disk full")
        return real(path, lines)
    monkeypatch.setattr(learner_main, "append_lines", append)
    lrn = learner_main.Learner()
    lrn.tick(now=1793000000)
    assert lrn.sensors_full is True and (tmp_path / "model_status.json").exists()
