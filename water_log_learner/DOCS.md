# Water log learner

Learns from the labels Richard sets on the Water page's water log card and
labels the house's water uses with a trained model (scikit-learn gradient
boosting). Every 30 seconds it labels new uses. It retrains a minute after a
new label and nightly at 3:30 AM, re-scoring itself on Richard's labels each
time (the card's footer shows the score).

Reads `/config/water_log/*.jsonl` and `labels.jsonl`. Writes only
`model_labels.jsonl` (append-only, a line only when a guess is new or changed)
and `model_status.json`. A guess below 60 % is stored as `null`, and the rules'
label stands. No options, no ports.

Design: ha-automations `docs/superpowers/specs/2026-11-17-water-log-labelling-learning-design.md`.
