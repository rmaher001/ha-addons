## 0.5.0

- Unknown rooms (rmaher001/ha-automations#712): a use the rules call unknown is placed in the kitchen, master or upstairs bathroom when anyone is in exactly one of those rooms as the water starts. Skipped, like faucet rooms, while a sink series has no points over a long read.

## 0.4.0

- Faucet rooms (rmaher001/ha-automations#712): a faucet is placed at the master vanity, upstairs vanity or kitchen when someone is at exactly one of those sinks as the water starts.

## 0.3.0

- Rooms (rmaher001/ha-automations#712): showers, baths and flushes placed in the master or upstairs bathroom from the master toilet's plug and each bathroom's motion and presence, written to `room_labels.jsonl`. Never trained on.

## 0.2.0

- Sensor labels (rmaher001/ha-automations#712): washer, dishwasher, shower and bath confirmed from the house's sensors in VictoriaMetrics, written to `sensor_labels.jsonl`, trained on at weight 10. Still no options: the VictoriaMetrics address is a constant in git.

## 0.1.1

- Regularised model: 0.1.0's unregularised settings diverged on the real history (84% agreement with the rules, confidently wrong).
- Sanity gate: a model agreeing with the rules on under 95% of unlabelled uses writes nothing and is marked rejected.

## 0.1.0

- First release (rmaher001/ha-addons#1).
