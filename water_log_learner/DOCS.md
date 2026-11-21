# Water log learner

Learns from the labels Richard sets on the Water page's water log card and
labels the house's water uses with a trained model (scikit-learn gradient
boosting). Every 30 seconds it labels new uses. It retrains a minute after a
new label and nightly at 3:30 AM, re-scoring itself on Richard's labels each
time (the card's footer shows the score).

It also asks the house's own sensors what each use was (washer status,
dishwasher plug power, the master and upstairs shower motion sensors), read
from VictoriaMetrics: all history on start and nightly, the last 48 hours every
10 minutes. What they confirm is written to `sensor_labels.jsonl`, shown on the
card with a ring, and taught to the model (Richard's labels still win). The
same pass places showers, baths and flushes in the master or upstairs bathroom
(a toilet seat's plug and each bathroom's motion and presence), and faucets at
the master vanity, upstairs vanity or kitchen sink (who is at which sink), in
`room_labels.jsonl`, the card's room column; rooms are never trained on.

Reads `/config/water_log/*.jsonl`, `labels.jsonl` and VictoriaMetrics. Writes
only `model_labels.jsonl`, `sensor_labels.jsonl` and `room_labels.jsonl`
(append-only, a line only when a label or room is new or changed) and `model_status.json`. A guess below 60 % is
stored as `null`, and the rules' label stands. A failed sensor read writes
nothing. No options, no ports: the VictoriaMetrics address is a constant in
`learner/main.py`.

Design: ha-automations `docs/superpowers/specs/2026-11-17-water-log-labelling-learning-design.md`.
