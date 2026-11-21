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
    assert set(seen) == {(100000 - 6 * 3600, 200000 + 6 * 3600)} and len(seen) == len(sensors.SERIES)


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
    for missing in ("top_load_washer", "master_bathroom_shower", "upstairs_bathroom_shower"):
        monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b, m=missing: [] if m in match else [(1.0, 1.0)])
        with pytest.raises(RuntimeError):
            sensors.evidence("http://vm.test", 0, 30 * 86400)
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: [] if "top_load_washer" in match else [(1.0, 1.0)])
    sensors.evidence("http://vm.test", 0, 2 * 86400)          # two days with no laundry is normal


@pytest.mark.requirement("water_log_sensors: an empty room series never fails the read (labels still flow); over a long read it marks the rooms unusable")
def test_empty_room_series_skips_rooms(monkeypatch):
    for missing in ("toilet_plug", "toilet_motion", "master_bathroom_presence", "upstairs_bathroom_motion_motion",
                    "upstairs_bathroom_presence"):
        monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b, m=missing: [] if m in match else [(1.0, 1.0)])
        ev = sensors.evidence("http://vm.test", 0, 30 * 86400)
        assert ev.rooms_ok is False and ev.sinks_ok is True and len(ev.empty) == 1
        assert sensors.evidence("http://vm.test", 0, 2 * 86400).rooms_ok is True     # a quiet two days is normal
    for missing in ("master_bedroom_vanity_presence", "master_bedroom_vanity_motion", "upstairs_vanity_presence",
                    "upstairs_vanity_motion", "hue_motion_01", "kitchen_camera"):
        monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b, m=missing: [] if m in match else [(1.0, 1.0)])
        ev = sensors.evidence("http://vm.test", 0, 30 * 86400)
        assert ev.sinks_ok is False and ev.rooms_ok is True and len(ev.empty) == 1   # a sink never stops the bathrooms
        assert sensors.evidence("http://vm.test", 0, 2 * 86400).sinks_ok is True
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: [(1.0, 1.0)])
    ev = sensors.evidence("http://vm.test", 0, 30 * 86400)
    assert ev.rooms_ok is True and ev.sinks_ok is True and ev.empty == ()


# ---- rooms (ha-automations#712, "Room column") ----------------------------------
# A plug on one toilet's seat bursts several hundred watts around a flush.
# Times and values here are invented.

from datetime import datetime, timezone  # noqa: E402

T0 = ts("2026-06-04T15:00:00+00:00")


def flush(t, gal=0.8):
    u = use(datetime.fromtimestamp(t, tz=timezone.utc).isoformat(), 15, gal, 3.4, 3.6)
    u["label"] = "flush"
    return u


def rooms_ev(toilet=(), master=(), upstairs=(), master_shower=(), upstairs_shower=()):
    return sensors.Evidence(washer=[], dishwasher=[], master=list(master_shower), upstairs=list(upstairs_shower),
                            toilet=list(toilet), master_near=(list(master), []), upstairs_near=([], list(upstairs)))


@pytest.mark.requirement("water_log_sensors: a flush with the toilet seat's plug bursting around it is the master bathroom")
def test_flush_with_plug_burst_is_master():
    ev = rooms_ev(toilet=[(T0 - 150, 2.0), (T0 - 120, 480.0), (T0 - 100, 25.0)])
    assert sensors.sensor_room(flush(T0), ev) == ("master", "toilet plug")


@pytest.mark.requirement("water_log_sensors: a plug burst already on at the window's start counts while its reading is under 4 minutes older than the window")
def test_plug_burst_carried_in():
    a = T0 - sensors.TOILET_BEFORE_S
    assert sensors.sensor_room(flush(T0), rooms_ev(toilet=[(a - 200, 480.0)])) == ("master", "toilet plug")
    assert sensors.sensor_room(flush(T0), rooms_ev(toilet=[(a - 300, 480.0)])) is None


@pytest.mark.requirement("water_log_sensors: master presence (the second master series) places a flush too")
def test_master_presence_places_a_flush():
    ev = sensors.Evidence(washer=[], dishwasher=[], master=[], upstairs=[], toilet=[],
                          master_near=([], [(T0 - 100, 1)]), upstairs_near=([], []))
    assert sensors.sensor_room(flush(T0), ev) == ("master", "master bathroom motion")


@pytest.mark.requirement("water_log_sensors: a flush-shaped use the sensors call washer or dishwasher gets no room")
def test_appliance_fill_gets_no_room():
    from datetime import datetime as dt
    t = WASHER_CHANGES[2] + 60
    u = use(dt.fromtimestamp(t, tz=timezone.utc).isoformat(), 60, 6.0, 6.0)
    u["label"] = "flush"
    ev = evidence()._replace(toilet=[(t - 60, 480.0)], master_near=([], []), upstairs_near=([], []))
    assert sensors.sensor_label(u, ev)[0] == "washer"
    assert sensors.sensor_room(u, ev) is None


@pytest.mark.requirement("water_log_sensors: the plug's idle seat warmer (under 50 W) is not a burst")
def test_idle_plug_is_not_a_burst():
    ev = rooms_ev(toilet=[(T0 - 100, 12.8), (T0 + 10, 30.0)])
    assert sensors.sensor_room(flush(T0), ev) is None


@pytest.mark.requirement("water_log_sensors: a plug burst long before the flush (over 4 minutes) does not count")
def test_old_plug_burst_does_not_count():
    ev = rooms_ev(toilet=[(T0 - 600, 480.0)])
    assert sensors.sensor_room(flush(T0), ev) is None
    ev = rooms_ev(toilet=[(T0 + 120, 480.0)])                    # and nor does one well after it
    assert sensors.sensor_room(flush(T0), ev) is None


@pytest.mark.requirement("water_log_sensors: master bathroom motion with no plug burst is still the master bathroom")
def test_flush_with_master_motion_is_master():
    ev = rooms_ev(master=[(T0 - 200, 1), (T0 + 20, 0)])
    assert sensors.sensor_room(flush(T0), ev) == ("master", "master bathroom motion")


@pytest.mark.requirement("water_log_sensors: a flush with only upstairs motion is the upstairs bathroom")
def test_flush_with_upstairs_motion_is_upstairs():
    ev = rooms_ev(upstairs=[(T0 - 90, 1), (T0 + 40, 0)])
    assert sensors.sensor_room(flush(T0, 1.8), ev) == ("upstairs", "upstairs bathroom motion")


@pytest.mark.requirement("water_log_sensors: someone in both bathrooms, or in neither, gives a flush no room")
def test_flush_room_needs_one_bathroom():
    both = rooms_ev(toilet=[(T0 - 60, 480.0)], upstairs=[(T0 - 60, 1)])
    assert sensors.sensor_room(flush(T0), both) is None
    assert sensors.sensor_room(flush(T0), rooms_ev()) is None


@pytest.mark.requirement("water_log_sensors: motion more than 5 minutes before a flush, or over a minute after, does not place it")
def test_flush_motion_window():
    assert sensors.sensor_room(flush(T0), rooms_ev(upstairs=[(T0 - 400, 1), (T0 - 380, 0)])) is None
    assert sensors.sensor_room(flush(T0), rooms_ev(upstairs=[(T0 + 90, 1)])) is None


@pytest.mark.requirement("water_log_sensors: a sensor-confirmed shower or bath takes the room of the shower motion that confirmed it")
def test_shower_and_bath_rooms():
    u = use(datetime.fromtimestamp(T0, tz=timezone.utc).isoformat(), 600, 20.0, 2.0)
    assert sensors.sensor_room(u, rooms_ev(master_shower=[(T0 - 30, 1)])) == ("master", "master shower motion")
    bath = use(u["start"], 1200, 50.0, 2.5)
    assert sensors.sensor_room(bath, rooms_ev(upstairs_shower=[(T0 - 30, 1)])) == ("upstairs", "upstairs shower motion")


@pytest.mark.requirement("water_log_sensors: uses the sensors do not confirm as shower or bath, and are not flushes, get no room")
def test_other_uses_get_no_room():
    sink = use(datetime.fromtimestamp(T0, tz=timezone.utc).isoformat(), 20, 0.4, 1.3)
    sink["label"] = "faucet"
    assert sensors.sensor_room(sink, rooms_ev(toilet=[(T0 - 60, 480.0)], master=[(T0 - 60, 1)])) is None
    short = use(sink["start"], 60, 2.5, 4.3)                    # in the shower room but too short to be a shower
    assert sensors.sensor_room(short, rooms_ev(upstairs_shower=[(T0 - 30, 1)])) is None


@pytest.mark.requirement("water_log_sensors: a tiny flush-labelled blip gets no room")
def test_tiny_gets_no_room():
    u = flush(T0, 0.05)
    assert sensors.sensor_room(u, rooms_ev(toilet=[(T0 - 60, 480.0)])) is None


@pytest.mark.requirement("water_log_sensors: room_all maps each placed use to (room, why)")
def test_room_all():
    ev = rooms_ev(toilet=[(T0 - 60, 480.0)])
    a, b = flush(T0), flush(T0 + 3600)
    assert sensors.room_all([a, b], ev) == {a["start"]: ("master", "toilet plug")}


@pytest.mark.requirement("water_log_sensors: room lines carry a room key, are written only when new or changed, and withdraw with room None")
def test_room_changed_lines():
    at = "2026-06-04T00:00:00+00:00"
    lines = sensors.changed_lines(["a", "b", "c"], {"a": ("master", "toilet plug"), "b": ("upstairs", "x")},
                                  {"b": "upstairs", "c": "master"}, at, key="room")
    assert lines == [{"start": "a", "room": "master", "why": "toilet plug", "by": "sensors", "at": at},
                     {"start": "c", "room": None, "why": None, "by": "sensors", "at": at}]


@pytest.mark.requirement("water_log_sensors: evidence carries the toilet plug and each bathroom's motion and presence")
def test_evidence_rooms(monkeypatch):
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: [(1.0, 7.0 if "toilet_plug" in match else 1.0)])
    ev = sensors.evidence("http://vm.test", 100000, 200000)
    assert ev.toilet == [(1.0, 7.0)]
    assert len(ev.master_near) == 2 and len(ev.upstairs_near) == 2


# ---- faucet rooms: which sink (master vanity, upstairs vanity, kitchen) -----------

def faucet(t, gal=0.4):
    u = use(datetime.fromtimestamp(t, tz=timezone.utc).isoformat(), 20, gal, 1.3, 1.5)
    u["label"] = "faucet"
    return u


def sinks_ev(master=(), upstairs=(), kitchen=()):
    return sensors.Evidence(washer=[], dishwasher=[], master=[], upstairs=[],
                            sinks=(("master", ([], list(master))), ("upstairs", (list(upstairs), [])),
                                   ("kitchen", ([], list(kitchen)))))


@pytest.mark.requirement("water_log_sensors: a faucet with someone at exactly one sink (master vanity, upstairs vanity or kitchen) is in that room")
def test_faucet_rooms():
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 - 30, 1)])) == ("kitchen", "kitchen motion")
    assert sensors.sensor_room(faucet(T0), sinks_ev(master=[(T0 - 10, 1)])) == ("master", "master vanity motion")
    assert sensors.sensor_room(faucet(T0), sinks_ev(upstairs=[(T0 + 10, 1)])) == ("upstairs", "upstairs vanity motion")


@pytest.mark.requirement("water_log_sensors: someone at two sinks, or at none, gives a faucet no room")
def test_faucet_room_needs_one_sink():
    assert sensors.sensor_room(faucet(T0), sinks_ev(master=[(T0 - 10, 1)], kitchen=[(T0 - 10, 1)])) is None
    assert sensors.sensor_room(faucet(T0), sinks_ev()) is None


@pytest.mark.requirement("water_log_sensors: sink motion counts from 1 minute before to 15 s after the water starts")
def test_faucet_window():
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 - 90, 1), (T0 - 70, 0)])) is None
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 + 30, 1)])) is None
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 - 55, 1), (T0 - 50, 0)])) == ("kitchen", "kitchen motion")
    # the edges themselves: 60 s before, 15 s after
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 - 59, 1), (T0 - 58, 0)])) == ("kitchen", "kitchen motion")
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 - 62, 1), (T0 - 61, 0)])) is None
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 + 14, 1)])) == ("kitchen", "kitchen motion")
    assert sensors.sensor_room(faucet(T0), sinks_ev(kitchen=[(T0 + 16, 1)])) is None


@pytest.mark.requirement("water_log_sensors: a sink sensor still on from up to 30 minutes before counts; older is a stuck sensor")
def test_sink_carry_over():
    assert sensors.sensor_room(faucet(T0), sinks_ev(master=[(T0 - 25 * 60, 1)])) == ("master", "master vanity motion")
    assert sensors.sensor_room(faucet(T0), sinks_ev(master=[(T0 - 35 * 60, 1)])) is None


@pytest.mark.requirement("water_log_sensors: a faucet-shaped use the sensors call dishwasher gets no room; flushes are not placed by sink sensors")
def test_sinks_place_only_faucets():
    slow = use("2026-06-03T08:09:37+00:00", 85, 1.09, 0.75, 0.78)
    slow["label"] = "faucet"
    t = ts(slow["start"])
    ev = evidence()._replace(sinks=(("kitchen", ([(t - 30, 1)],)),))
    assert sensors.sensor_label(slow, ev)[0] == "dishwasher"
    assert sensors.sensor_room(slow, ev) is None
    assert sensors.sensor_room(flush(T0), sinks_ev(kitchen=[(T0 - 30, 1)])) is None


@pytest.mark.requirement("water_log_sensors: evidence carries each sink's motion and presence series")
def test_evidence_sinks(monkeypatch):
    monkeypatch.setattr(sensors, "fetch", lambda url, match, a, b: [(1.0, 1.0)])
    ev = sensors.evidence("http://vm.test", 100000, 200000)
    assert [room for room, _series in ev.sinks] == ["master", "upstairs", "kitchen"]
    assert all(len(series) == 2 for _room, series in ev.sinks)
