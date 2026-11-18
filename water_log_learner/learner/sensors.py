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
"""
import bisect
import json
import urllib.parse
import urllib.request
from collections import namedtuple
from datetime import datetime

Evidence = namedtuple("Evidence", "washer dishwasher master upstairs")

SERIES = {
    "washer": '{__name__="units_state",entity_id="top_load_washer_current_status"}',
    "dishwasher": '{__name__="W_value",entity_id="kitchen_dishwasher_plug_power"}',
    "master": '{__name__="units_value",entity_id="master_bathroom_shower_motion_occupancy"}',
    "upstairs": '{__name__="units_value",entity_id="upstairs_bathroom_shower_motion_occupancy"}',
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


def _active(points, a, b):
    """Motion on at any moment in [a, b]: an "on" sample inside, or the last
    sample before `a` was "on" (the sensor reports changes, not a heartbeat)
    and no older than MOTION_STALE_S."""
    times = [t for t, _v in points]
    i = bisect.bisect_left(times, a)
    if i > 0 and points[i - 1][1] >= 0.5 and a - points[i - 1][0] <= MOTION_STALE_S:
        return True
    j = i
    while j < len(points) and points[j][0] <= b:
        if points[j][1] >= 0.5:
            return True
        j += 1
    return False


def sensor_label(u, ev):
    """(label, why) when the sensors confirm one explanation, else None."""
    gal, dur, med = float(u["gallons"]), float(u["duration_s"]), float(u["median_gpm"])
    if gal < MIN_GALLONS or u.get("label") == "tiny":
        return None
    t = epoch(u["start"])
    window = (t - MOTION_BEFORE_S, t + MOTION_AFTER_S)
    rooms = [room for room in ("master", "upstairs") if _active(getattr(ev, room), *window)]
    if len(rooms) > 1:
        return None
    washer = _inside(t, ev.washer) and gal >= WASHER_MIN_GAL and med >= WASHER_MIN_GPM
    dish = (_inside(t, ev.dishwasher) and med <= DISH_MAX_GPM and dur <= DISH_MAX_S
            and float(u["peak_gpm"]) <= DISH_MAX_PEAK)
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


def changed_lines(starts, current, previous, at):
    """Append-only lines: a label that is new or different, and a withdrawal
    (label None) for one the sensors no longer support."""
    out, seen = [], set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        label, why = current.get(start, (None, None))
        if previous.get(start) == label:      # also skips "no label" for a use never labelled
            continue
        out.append({"start": start, "label": label, "why": why, "by": "sensors", "at": at})
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
    if b - a > LONG_READ_S:
        empty = [k for k, pts in got.items() if not pts]
        if empty:
            raise RuntimeError(f"no points for {', '.join(empty)} over {int((b - a) / 86400)} days")
    return Evidence(washer=cycles([t for t, _v in got["washer"]]), dishwasher=runs(got["dishwasher"]),
                    master=got["master"], upstairs=got["upstairs"])
