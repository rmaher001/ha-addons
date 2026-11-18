"""Sensor labels: the house's own sensors confirm what a water use was
(rmaher001/ha-automations#712, plan section "Sensor labels"). The fixtures keep
the SHAPE of one real evening and morning (gaps, rates, sizes, which sensor
fired when relative to the water); the dates and times of day are shifted by a
constant so no real household timing is published."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "water_log_learner"))

from learner import sensors  # noqa: E402  (a missing module must FAIL, not skip)


def use(start, dur, gal, med, peak=None):
    return {"start": start, "end": start, "duration_s": dur, "gallons": gal, "median_gpm": med,
            "peak_gpm": peak if peak is not None else med, "label": "unknown", "rules_version": 2,
            "source": "live"}


def ts(iso):
    return sensors.epoch(iso)


# One load's washer status changes, initial to power off. VictoriaMetrics keeps only the times.
WASHER_CHANGES = [1780477528, 1780477536, 1780477612, 1780479266, 1780480304, 1780481318, 1780481347]

# Dishwasher plug power: on, a 2-min dip, heating, off after about 3 h 20 min.
DISHWASHER = [(ts("2026-06-03T07:57:00+00:00"), 3.6), (ts("2026-06-03T08:08:42+00:00"), 7.2),
              (ts("2026-06-03T08:29:47+00:00"), 4.6), (ts("2026-06-03T08:31:47+00:00"), 16.5),
              (ts("2026-06-03T10:00:32+00:00"), 3.9), (ts("2026-06-03T10:02:33+00:00"), 17.2),
              (ts("2026-06-03T11:30:46+00:00"), 0.8),
              # a standby blip: 13 minutes, not a run
              (ts("2026-06-03T04:56:38+00:00"), 7.2), (ts("2026-06-03T05:10:05+00:00"), 3.8)]

MASTER = [(ts("2026-06-03T08:50:10+00:00"), 1), (ts("2026-06-03T08:59:13+00:00"), 0),
          (ts("2026-06-03T18:02:24+00:00"), 1), (ts("2026-06-03T18:13:03+00:00"), 0)]
UPSTAIRS = [(ts("2026-06-03T07:10:12+00:00"), 1), (ts("2026-06-03T07:43:37+00:00"), 0)]


def evidence():
    return sensors.Evidence(washer=sensors.cycles(WASHER_CHANGES),
                            dishwasher=sensors.runs(sorted(DISHWASHER)),
                            master=MASTER, upstairs=UPSTAIRS)


@pytest.mark.requirement("water_log_sensors: a cluster of washer status changes is one wash cycle; a lone change is not")
def test_washer_cycles():
    got = sensors.cycles(WASHER_CHANGES + [1780496620])
    assert got == [(1780477528, 1780481347)]


@pytest.mark.requirement("water_log_sensors: a dishwasher run is power above 5 W for 45 min or more, short dips merged; a standby blip is not")
def test_dishwasher_runs():
    got = sensors.runs(sorted(DISHWASHER))
    assert got == [(ts("2026-06-03T08:08:42+00:00"), ts("2026-06-03T11:30:46+00:00"))]


@pytest.mark.requirement("water_log_sensors: the washer's three fills that night are washer; a toilet and a sink during the load are not")
def test_washer_uses():
    ev = evidence()
    assert sensors.sensor_label(use("2026-06-03T09:06:57+00:00", 160, 18.07, 6.56, 8.7), ev)[0] == "washer"
    assert sensors.sensor_label(use("2026-06-03T09:10:02+00:00", 65, 6.67, 6.44, 7.31), ev)[0] == "washer"
    assert sensors.sensor_label(use("2026-06-03T09:40:47+00:00", 335, 21.92, 4.02, 4.96), ev)[0] == "washer"
    assert sensors.sensor_label(use("2026-06-03T09:55:02+00:00", 10, 0.77, 3.53, 3.6), ev) is None   # flush
    assert sensors.sensor_label(use("2026-06-03T09:14:07+00:00", 5, 0.15, 1.46), ev) is None        # sink


@pytest.mark.requirement("water_log_sensors: slow fills during a dishwasher run are dishwasher; a sink at faucet speed is not")
def test_dishwasher_uses():
    ev = evidence()
    assert sensors.sensor_label(use("2026-06-03T08:09:37+00:00", 85, 1.09, 0.75, 0.78), ev)[0] == "dishwasher"
    assert sensors.sensor_label(use("2026-06-03T08:32:52+00:00", 65, 0.9, 0.7, 0.71), ev)[0] == "dishwasher"
    assert sensors.sensor_label(use("2026-06-03T10:48:47+00:00", 5, 0.17, 0.27), ev)[0] == "dishwasher"
    assert sensors.sensor_label(use("2026-06-03T08:45:02+00:00", 8, 0.2, 1.34), ev) is None          # sink
    assert sensors.sensor_label(use("2026-06-03T13:17:00+00:00", 60, 0.8, 0.7), ev) is None          # after the run


@pytest.mark.requirement("water_log_sensors: a use both appliances could explain gets no label")
def test_both_appliances_is_ambiguous():
    ev = evidence()
    # inside the washer cycle AND the dishwasher run, too small for a washer fill
    assert sensors.sensor_label(use("2026-06-03T09:06:42+00:00", 5, 0.2, 0.88), ev) is None


@pytest.mark.requirement("water_log_sensors: master shower motion at the start makes it a shower at any flow (1.4 and 4 gal/min both seen)")
def test_master_showers():
    ev = evidence()
    assert sensors.sensor_label(use("2026-06-03T08:50:12+00:00", 535, 12.92, 1.36, 4.65), ev) == ("shower", "master shower motion")
    assert sensors.sensor_label(use("2026-06-03T18:02:27+00:00", 600, 28.38, 3.95, 8.41), ev) == ("shower", "master shower motion")


@pytest.mark.requirement("water_log_sensors: upstairs shower motion means bath when it is big (>= 40 gal, >= 15 min), else shower")
def test_upstairs_bath_or_shower():
    ev = evidence()
    assert sensors.sensor_label(use("2026-06-03T07:12:07+00:00", 1685, 91.5, 3.22, 6.7), ev) == ("bath", "upstairs shower motion")
    assert sensors.sensor_label(use("2026-06-03T07:12:07+00:00", 480, 14.0, 1.8, 2.5), ev) == ("shower", "upstairs shower motion")


@pytest.mark.requirement("water_log_sensors: motion must be near the START; short or small uses in the bathroom are not showers")
def test_shower_needs_motion_at_the_start_and_size():
    ev = evidence()
    # the sink in the master bath while the shower sensor is on: too short
    assert sensors.sensor_label(use("2026-06-03T08:51:00+00:00", 40, 0.9, 1.4), ev) is None
    # a long use with no motion near its start
    assert sensors.sensor_label(use("2026-06-03T15:17:00+00:00", 600, 20.0, 2.0), ev) is None
    # motion 4 minutes before the water: too early
    late = sensors.Evidence(washer=[], dishwasher=[], master=[(ts("2026-06-03T15:17:00+00:00"), 1),
                                                              (ts("2026-06-03T15:17:30+00:00"), 0)], upstairs=[])
    assert sensors.sensor_label(use("2026-06-03T15:21:00+00:00", 600, 20.0, 2.0), late) is None


@pytest.mark.requirement("water_log_sensors: motion already on from before the window counts (the sensor reports changes, not a heartbeat)")
def test_motion_on_since_before():
    ev = sensors.Evidence(washer=[], dishwasher=[], master=[(ts("2026-06-03T15:17:00+00:00"), 1)], upstairs=[])
    assert sensors.sensor_label(use("2026-06-03T15:27:00+00:00", 600, 20.0, 2.0), ev)[0] == "shower"


@pytest.mark.requirement("water_log_sensors: both bathrooms' motion at once gives no label")
def test_both_bathrooms_is_ambiguous():
    ev = sensors.Evidence(washer=[], dishwasher=[], master=MASTER,
                          upstairs=[(ts("2026-06-03T18:02:00+00:00"), 1), (ts("2026-06-03T18:07:00+00:00"), 0)])
    assert sensors.sensor_label(use("2026-06-03T18:02:27+00:00", 600, 28.38, 3.95, 8.41), ev) is None


@pytest.mark.requirement("water_log_sensors: tiny uses are never labelled")
def test_tiny_never():
    ev = evidence()
    assert sensors.sensor_label(use("2026-06-03T08:10:00+00:00", 5, 0.05, 0.7), ev) is None


@pytest.mark.requirement("water_log_sensors: the VictoriaMetrics export is parsed into sorted (seconds, value) points across series; junk lines are skipped")
def test_parse_export():
    text = ('{"metric":{"__name__":"units_value"},"values":[1,0],"timestamps":[2000000,1000000]}\n'
            'not json\n'
            '{"metric":{},"values":[1],"timestamps":[1500000]}\n'
            '{"metric":{},"values":[1,2],"timestamps":[1]}\n')
    assert sensors.parse_export(text) == [(1000.0, 0.0), (1500.0, 1.0), (2000.0, 1.0)]


@pytest.mark.requirement("water_log_sensors: changed_lines writes a label only when it is new or different, and withdraws one the sensors no longer support")
def test_changed_lines():
    now = "2026-06-03T19:17:00+00:00"
    previous = {"a": "washer", "b": "shower", "c": None}
    current = {"a": ("washer", "washer cycle"), "d": ("bath", "upstairs shower motion")}
    lines = sensors.changed_lines(["a", "b", "c", "d"], current, previous, now)
    assert lines == [{"start": "b", "label": None, "why": None, "by": "sensors", "at": now},
                     {"start": "d", "label": "bath", "why": "upstairs shower motion", "by": "sensors", "at": now}]


@pytest.mark.requirement("water_log_sensors: a toilet flush during a dishwasher run (low median, peak above 3 gal/min) is not dishwasher")
def test_flush_during_dishwasher_run():
    ev = evidence()
    assert sensors.sensor_label(use("2026-06-03T08:37:00+00:00", 20, 0.79, 0.91, 3.35), ev) is None


@pytest.mark.requirement("water_log_sensors: a use the rules call tiny is never labelled, whatever its rounding")
def test_rule_tiny_never():
    ev = evidence()
    u = use("2026-06-03T08:10:00+00:00", 5, 0.1, 0.7)
    u["label"] = "tiny"
    assert sensors.sensor_label(u, ev) is None


@pytest.mark.requirement("water_log_sensors: a washer-shaped fill inside a wash cycle with shower motion nearby gets no label (two explanations)")
def test_shower_motion_during_a_washer_fill_is_ambiguous():
    ev = evidence()
    from datetime import datetime, timezone
    fill = use(datetime.fromtimestamp(WASHER_CHANGES[2] + 60, tz=timezone.utc).isoformat(), 200, 18.0, 5.4)
    t = sensors.epoch(fill["start"])
    busy = ev._replace(master=[(t - 120, 1), (t + 30, 0)])
    assert sensors.sensor_label(fill, ev)[0] == "washer"
    assert sensors.sensor_label(fill, busy) is None


@pytest.mark.requirement("water_log_sensors: a motion 'on' left over from more than 30 minutes before does not count (a stuck sensor), so the full and 48-hour reads agree")
def test_stale_motion_does_not_count():
    t = ts("2026-06-03T12:00:00+00:00")
    stale = sensors.Evidence(washer=[], dishwasher=[], master=[(t - 3600, 1)], upstairs=[])
    assert sensors.sensor_label(use("2026-06-03T12:00:00+00:00", 600, 20.0, 2.0), stale) is None
    fresh = sensors.Evidence(washer=[], dishwasher=[], master=[(t - 600, 1)], upstairs=[])
    assert sensors.sensor_label(use("2026-06-03T12:00:00+00:00", 600, 20.0, 2.0), fresh)[0] == "shower"


@pytest.mark.requirement("water_log_sensors: a dishwasher run still going at the end of the data counts up to its last reading")
def test_run_in_progress():
    assert sensors.runs([(0, 1), (60, 50), (3600, 60)]) == [(60, 3600)]
    assert sensors.runs([(0, 1), (60, 50), (600, 60)]) == []          # too short so far


@pytest.mark.requirement("water_log_sensors: a slow fill inside a wash cycle is not washer (washer fills run >= 3 gal/min)")
def test_washer_needs_speed():
    ev = evidence()
    t = WASHER_CHANGES[2] + 60
    from datetime import datetime, timezone
    assert sensors.sensor_label(use(datetime.fromtimestamp(t, tz=timezone.utc).isoformat(), 200, 6.0, 1.8), ev) is None


@pytest.mark.requirement("water_log_sensors: a washer-sized fill with motion in both bathrooms gets no label")
def test_both_bathrooms_blocks_appliances_too():
    from datetime import datetime, timezone
    t = WASHER_CHANGES[2] + 60
    both = evidence()._replace(master=[(t - 60, 1)], upstairs=[(t - 60, 1)])
    assert sensors.sensor_label(use(datetime.fromtimestamp(t, tz=timezone.utc).isoformat(), 60, 6.0, 6.0), both) is None


@pytest.mark.requirement("water_log_sensors: the master bathroom has no tub: a bath-sized use there is still a shower")
def test_master_is_never_bath():
    t = ts("2026-06-03T12:00:00+00:00")
    ev = sensors.Evidence(washer=[], dishwasher=[], master=[(t - 30, 1)], upstairs=[])
    assert sensors.sensor_label(use("2026-06-03T12:00:00+00:00", 1700, 90.0, 3.2), ev)[0] == "shower"


@pytest.mark.requirement("water_log_sensors: a start listed twice gets one line")
def test_changed_lines_dedupes():
    lines = sensors.changed_lines(["a", "a"], {"a": ("washer", "washer cycle")}, {}, "2026-06-03T00:00:00+00:00")
    assert len(lines) == 1


@pytest.mark.requirement("water_log_sensors: the read reaches 6 hours past each end of the uses, so a whole dishwasher run is seen")
def test_evidence_margin(monkeypatch):
    seen = []
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: seen.append((a, b)) or [(1.0, 1.0)])
    sensors.evidence("http://vm.test", 100000, 200000)
    assert set(seen) == {(100000 - 6 * 3600, 200000 + 6 * 3600)} and len(seen) == 4


@pytest.mark.requirement("water_log_sensors: fetch posts the series and range to /api/v1/export and parses the answer")
def test_fetch(monkeypatch):
    got = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"metric":{},"values":[1],"timestamps":[5000]}\n'

    def fake_urlopen(url, data=None, timeout=None):
        got.update(url=url, data=data.decode(), timeout=timeout)
        return Resp()
    monkeypatch.setattr(sensors.urllib.request, "urlopen", fake_urlopen)
    assert sensors.fetch("http://vm.test:8428/", '{entity_id="x"}', 10, 20.7) == [(5.0, 1.0)]
    assert got["url"] == "http://vm.test:8428/api/v1/export"
    assert "match%5B%5D=%7Bentity_id%3D%22x%22%7D" in got["data"] and "start=10" in got["data"] and "end=20" in got["data"]


@pytest.mark.requirement("water_log_sensors: over a long range every series must answer; a missing washer or motion series fails the read instead of withdrawing labels")
def test_long_read_needs_every_series(monkeypatch):
    for missing in ("top_load_washer", "master_bathroom", "upstairs_bathroom"):
        monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b, m=missing: [] if m in match else [(1.0, 1.0)])
        with pytest.raises(RuntimeError):
            sensors.evidence("http://vm.test", 0, 30 * 86400)
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: [] if "top_load_washer" in match else [(1.0, 1.0)])
    sensors.evidence("http://vm.test", 0, 2 * 86400)          # two days with no laundry is normal
