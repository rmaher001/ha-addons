"""Sensor labels: the house's own sensors confirm what a water use was
(rmaher001/ha-automations#712, plan section "Sensor labels").

Pure functions over sensor history read from VictoriaMetrics, plus the one
read (`fetch`). A label is given only when exactly one explanation fits; any
doubt is no label, and the rules or the model stand.

Evidence, all kept in VictoriaMetrics since 2026-03-16:
  * washer - `top_load_washer_current_status`. The store keeps only WHEN the
    status changed (the text value is lost), so a wash cycle is a cluster of
    status changes;
  * dishwasher - its plug's power;
  * showers - the shower motion sensor in the master and upstairs bathrooms.
    The master bathroom has no tub; the upstairs one is shower or bath by size
    (Richard, 2026-11-18).

Rooms (plan section "Room column"): which bathroom a shower, bath or flush was
in, written apart from the labels and never trained on. A flush is placed by a
plug on the master toilet's seat, whose heater bursts several hundred watts
around a flush, and by each bathroom's motion and presence sensors. A faucet
is placed at the one sink - master vanity, upstairs vanity or kitchen - where
someone is when the water starts. A use the rules call unknown is placed in
the one room - kitchen, master or upstairs bathroom - anyone is in then.
"""
import bisect
import json
import urllib.parse
import urllib.request
from collections import namedtuple
from datetime import datetime

# master/upstairs: shower motion. toilet: the master toilet plug's watts.
# master_near/upstairs_near: every other motion or presence series per bathroom.
# sinks: ((room, (series, ...)), ...) - who is at each sink.
# Over a long read, a series with no points at all did not answer: rooms_ok
# False (a bathroom series) places no rooms; sinks_ok False (a sink series)
# places no faucet rooms. `empty` names them.
Evidence = namedtuple("Evidence", "washer dishwasher master upstairs toilet master_near upstairs_near sinks "
                      "rooms_ok sinks_ok empty", defaults=((), (), (), (), True, True, ()))
SINKS = (("master", ("master_vanity_presence", "master_vanity_motion")),
         ("upstairs", ("upstairs_vanity_presence", "upstairs_vanity_motion")),
         ("kitchen", ("kitchen_motion", "kitchen_camera")))
BATHROOM_SERIES = ("toilet", "master_toilet", "master_presence", "upstairs_motion", "upstairs_presence")
SINK_SERIES = tuple(k for _room, keys in SINKS for k in keys)
ROOM_SERIES = BATHROOM_SERIES + SINK_SERIES
SINK_WHY = {"master": "master vanity motion", "upstairs": "upstairs vanity motion", "kitchen": "kitchen motion"}

SERIES = {
    "washer": '{__name__="units_state",entity_id="top_load_washer_current_status"}',
    "dishwasher": '{__name__="W_value",entity_id="kitchen_dishwasher_plug_power"}',
    "master": '{__name__="units_value",entity_id="master_bathroom_shower_motion_occupancy"}',
    "upstairs": '{__name__="units_value",entity_id="upstairs_bathroom_shower_motion_occupancy"}',
    "toilet": '{__name__="W_value",entity_id="master_bathroom_toilet_plug_power"}',
    "master_toilet": '{__name__="units_value",entity_id="master_bathroom_toilet_motion_occupancy"}',
    "master_presence": '{__name__="units_value",entity_id="master_bathroom_presence_filtered"}',
    "upstairs_motion": '{__name__="units_value",entity_id="upstairs_bathroom_motion_motion"}',
    "upstairs_presence": '{__name__="units_value",entity_id="upstairs_bathroom_presence_filtered"}',
    "master_vanity_presence": '{__name__="units_value",entity_id="master_bedroom_vanity_presence_filtered"}',
    "master_vanity_motion": '{__name__="units_value",entity_id="master_bedroom_vanity_motion_occupancy"}',
    "upstairs_vanity_presence": '{__name__="units_value",entity_id="upstairs_vanity_presence_filtered"}',
    "upstairs_vanity_motion": '{__name__="units_value",entity_id="upstairs_vanity_motion_motion"}',
    "kitchen_motion": '{__name__="units_value",entity_id="hue_motion_01_motion"}',
    "kitchen_camera": '{__name__="units_value",entity_id="kitchen_kitchen_camera_smart_occupancy_sensor_occupancy"}',
}

MARGIN_S = 6 * 3600
LONG_READ_S = 7 * 86400     # a read this long must find every series
MIN_GALLONS = 0.1            # tiny uses are never labelled

# Washer: >= 4 status changes, none more than 45 min apart, spanning >= 20 min.
# Its fills are fast (4-7 gal/min) and big.
CYCLE_GAP_S, CYCLE_MIN_CHANGES, CYCLE_MIN_S = 45 * 60, 4, 20 * 60
WASHER_MIN_GAL, WASHER_MIN_GPM = 2.0, 3.0

# Dishwasher: plug above 5 W (standby is ~4 W), dips under 5 min merged, >= 45 min.
# Its fills are slow (~0.7 gal/min) and short; a kitchen faucet runs ~1.4.
RUN_W, RUN_GAP_S, RUN_MIN_S = 5.0, 5 * 60, 45 * 60
# A toilet flush in the middle of a run has a low median too, but peaks above
# 3 gal/min; 95% of dishwasher fills peak under 2.1 (8 months, 2026-11-18).
DISH_MAX_GPM, DISH_MAX_PEAK, DISH_MAX_S = 1.0, 2.0, 180

# Shower motion from 3 min before to 2 min after the water starts; a use of at
# least 3 min and 3 gal (a sink in the same bathroom is shorter).
MOTION_BEFORE_S, MOTION_AFTER_S = 180, 120
# An "on" reported longer ago than this before the window is a stuck sensor,
# not someone in the shower. Well inside MARGIN_S, so the 48-hour read and the
# all-history read always see the same thing.
MOTION_STALE_S = 30 * 60
SHOWER_MIN_S, SHOWER_MIN_GAL = 180, 3.0
BATH_MIN_GAL, BATH_MIN_S = 40.0, 15 * 60

# Flush rooms. The plug idles well under 50 W (seat warmer); its heater bursts
# start a few minutes before the flush and end soon after. Bathroom motion or presence
# counts from 5 min before to 1 min after the water starts.
TOILET_W, TOILET_BEFORE_S, TOILET_AFTER_S = 50.0, 240, 60
NEAR_BEFORE_S, NEAR_AFTER_S = 300, 60
# Faucets: someone at the sink from 1 min before to 15 s after the water starts.
# The kitchen's sensors are on far more often than the vanities', so two sinks
# at once is common and gives no room.
SINK_BEFORE_S, SINK_AFTER_S = 60, 15


def epoch(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def cycles(times):
    """[(first, last)] of each wash cycle in a list of status-change times."""
    out, group = [], []
    for t in sorted(times):
        if group and t - group[-1] > CYCLE_GAP_S:
            out.append(group)
            group = []
        group.append(t)
    if group:
        out.append(group)
    return [(g[0], g[-1]) for g in out if len(g) >= CYCLE_MIN_CHANGES and g[-1] - g[0] >= CYCLE_MIN_S]


def runs(points):
    """[(on, off)] of each dishwasher run in sorted (time, watts) points."""
    spans, on = [], None
    for t, w in points:
        if w > RUN_W and on is None:
            on = t
        elif w <= RUN_W and on is not None:
            spans.append([on, t])
            on = None
    if on is not None and points:
        spans.append([on, points[-1][0]])     # still running: count it up to its last reading
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < RUN_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= RUN_MIN_S]


def _inside(t, spans):
    return any(a <= t <= b for a, b in spans)


def _active(points, a, b, above=0.5, stale=MOTION_STALE_S):
    """On at any moment in [a, b]: a sample at or above `above` inside, or the
    last sample before `a` was (the sensors report changes, not a heartbeat)
    and no older than `stale`."""
    times = [t for t, _v in points]
    i = bisect.bisect_left(times, a)
    if i > 0 and points[i - 1][1] >= above and a - points[i - 1][0] <= stale:
        return True
    j = i
    while j < len(points) and points[j][0] <= b:
        if points[j][1] >= above:
            return True
        j += 1
    return False


def _shower_rooms(u, ev):
    t = epoch(u["start"])
    return [room for room in ("master", "upstairs")
            if _active(getattr(ev, room), t - MOTION_BEFORE_S, t + MOTION_AFTER_S)]


def _appliances(u, ev):
    """(washer fits, dishwasher fits) for a use, by the appliance's run and the fill's speed."""
    gal, dur, med = float(u["gallons"]), float(u["duration_s"]), float(u["median_gpm"])
    t = epoch(u["start"])
    washer = _inside(t, ev.washer) and gal >= WASHER_MIN_GAL and med >= WASHER_MIN_GPM
    dish = (_inside(t, ev.dishwasher) and med <= DISH_MAX_GPM and dur <= DISH_MAX_S
            and float(u["peak_gpm"]) <= DISH_MAX_PEAK)
    return washer, dish


def sensor_label(u, ev):
    """(label, why) when the sensors confirm one explanation, else None."""
    gal, dur, med = float(u["gallons"]), float(u["duration_s"]), float(u["median_gpm"])
    if gal < MIN_GALLONS or u.get("label") == "tiny":
        return None
    t = epoch(u["start"])
    rooms = _shower_rooms(u, ev)
    if len(rooms) > 1:
        return None
    washer, dish = _appliances(u, ev)
    if dur >= SHOWER_MIN_S and gal >= SHOWER_MIN_GAL and rooms:
        if washer or dish:
            return None                        # an appliance fits too: two explanations
        why = f"{rooms[0]} shower motion"
        if rooms[0] == "upstairs" and gal >= BATH_MIN_GAL and dur >= BATH_MIN_S:
            return ("bath", why)
        return ("shower", why)
    # A washer fill (>= 3 gal/min) and a dishwasher fill (<= 1) cannot both fit;
    # a slow fill during a wash cycle could be either appliance, so it is none.
    if washer:
        return ("washer", "washer cycle")
    if dish and not _inside(t, ev.washer):
        return ("dishwasher", "dishwasher running")
    return None


def label_all(uses, ev):
    """start -> (label, why) for every use the sensors confirm."""
    out = {}
    for u in uses:
        got = sensor_label(u, ev)
        if got:
            out[u["start"]] = got
    return out


def sensor_room(u, ev):
    """(room, why) - "master", "upstairs" or (faucets and unknowns only)
    "kitchen" - for a sensor-confirmed shower or bath, a flush, a faucet or an
    unknown use; None when no room, or more than one, fits."""
    if float(u["gallons"]) < MIN_GALLONS or u.get("label") == "tiny":
        return None
    got = sensor_label(u, ev)
    if got and got[0] in ("shower", "bath"):
        return (_shower_rooms(u, ev)[0], got[1])
    if got:                                    # an appliance fill is in no bathroom
        return None
    t = epoch(u["start"])
    if u.get("label") == "faucet":
        here = [room for room, series in ev.sinks
                if any(_active(p, t - SINK_BEFORE_S, t + SINK_AFTER_S) for p in series)]
        return (here[0], SINK_WHY[here[0]]) if len(here) == 1 else None
    if u.get("label") == "unknown":
        # An appliance fill is in no bathroom, even when only "maybe" one.
        return None if any(_appliances(u, ev)) else _unknown_room(u, ev)
    if u.get("label") != "flush":
        return None
    a, b = t - NEAR_BEFORE_S, t + NEAR_AFTER_S
    plug = _active(ev.toilet, t - TOILET_BEFORE_S, t + TOILET_AFTER_S, above=TOILET_W, stale=TOILET_BEFORE_S)
    master = plug or any(_active(p, a, b) for p in ev.master_near)
    upstairs = any(_active(p, a, b) for p in ev.upstairs_near)
    if master == upstairs:
        return None
    if master:
        return ("master", "toilet plug" if plug else "master bathroom motion")
    return ("upstairs", "upstairs bathroom motion")


UNKNOWN_WHY = {"master": "only the master bathroom had someone",
               "upstairs": "only the upstairs bathroom had someone",
               "kitchen": "only the kitchen had someone"}


def _unknown_room(u, ev):
    """(room, why) for a use the rules could not name: the one room - kitchen,
    master or upstairs bathroom - where anyone was when the water started.
    Each sink counts over the sink window; a bathroom's shower motion, other
    motion and presence (and the master toilet plug) over the bathroom window.
    Nobody seen, or people in two rooms, is no room (Richard, 2026-11-21)."""
    t = epoch(u["start"])
    a, b = t - NEAR_BEFORE_S, t + NEAR_AFTER_S
    here = {room for room, series in ev.sinks
            if any(_active(p, t - SINK_BEFORE_S, t + SINK_AFTER_S) for p in series)}
    if (_active(ev.toilet, t - TOILET_BEFORE_S, t + TOILET_AFTER_S, above=TOILET_W, stale=TOILET_BEFORE_S)
            or _active(ev.master, a, b) or any(_active(p, a, b) for p in ev.master_near)):
        here.add("master")
    if _active(ev.upstairs, a, b) or any(_active(p, a, b) for p in ev.upstairs_near):
        here.add("upstairs")
    if len(here) != 1:
        return None
    room = here.pop()
    return (room, UNKNOWN_WHY[room])


def room_all(uses, ev):
    """start -> (room, why) for every use the sensors place."""
    out = {}
    for u in uses:
        got = sensor_room(u, ev)
        if got:
            out[u["start"]] = got
    return out


def changed_lines(starts, current, previous, at, key="label"):
    """Append-only lines: a `key` value (label or room) that is new or
    different, and a withdrawal (None) for one the sensors no longer support."""
    out, seen = [], set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        value, why = current.get(start, (None, None))
        if previous.get(start) == value:      # also skips "nothing" for a use never given one
            continue
        out.append({"start": start, key: value, "why": why, "by": "sensors", "at": at})
    return out


# ---- the one read ------------------------------------------------------------

def parse_export(text):
    """Sorted (seconds, value) points from a VictoriaMetrics /api/v1/export body."""
    pts = []
    for line in text.splitlines():
        try:
            d = json.loads(line)
            ts, vs = d["timestamps"], d["values"]
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(ts, list) or not isinstance(vs, list) or len(ts) != len(vs):
            continue
        pts.extend((t / 1000.0, float(v)) for t, v in zip(ts, vs) if v is not None)
    pts.sort()
    return pts


def fetch(vm_url, match, start, end, timeout=30):
    """Points of one series between two epoch seconds. Raises on any failure:
    no data is never mistaken for "nothing happened"."""
    body = urllib.parse.urlencode({"match[]": match, "start": int(start), "end": int(end)}).encode()
    with urllib.request.urlopen(vm_url.rstrip("/") + "/api/v1/export", data=body, timeout=timeout) as r:
        return parse_export(r.read().decode("utf-8", "replace"))


def evidence(vm_url, start, end):
    """Evidence for uses starting in [start, end], with MARGIN_S each side: a
    dishwasher run lasts 3.5 hours and must be seen whole from either end."""
    a, b = start - MARGIN_S, end + MARGIN_S
    got = dict((k, fetch(vm_url, m, a, b)) for k, m in SERIES.items())
    # The plug reports every few minutes, day and night: none at all means the
    # store did not really answer, and "no evidence" must never read as "no
    # shower" (that would withdraw labels).
    if not got["dishwasher"]:
        raise RuntimeError("no dishwasher plug points: VictoriaMetrics returned nothing")
    # Over a long read every series has points (laundry and showers happen
    # weekly): an empty one is a renamed or missing series, and reading it as
    # "nothing happened" would withdraw every label of that kind.
    # Room series never fail the read (labels must still flow): an empty one
    # over a long read only stops rooms being placed or withdrawn.
    empty = ()
    if b - a > LONG_READ_S:
        missing = [k for k, pts in got.items() if not pts and k not in ROOM_SERIES]
        if missing:
            raise RuntimeError(f"no points for {', '.join(missing)} over {int((b - a) / 86400)} days")
        empty = tuple(k for k in ROOM_SERIES if not got[k])
    return Evidence(washer=cycles([t for t, _v in got["washer"]]), dishwasher=runs(got["dishwasher"]),
                    master=got["master"], upstairs=got["upstairs"], toilet=got["toilet"],
                    master_near=(got["master_toilet"], got["master_presence"]),
                    upstairs_near=(got["upstairs_motion"], got["upstairs_presence"]),
                    sinks=tuple((room, tuple(got[k] for k in keys)) for room, keys in SINKS),
                    rooms_ok=not set(empty) & set(BATHROOM_SERIES), sinks_ok=not set(empty) & set(SINK_SERIES),
                    empty=empty)
