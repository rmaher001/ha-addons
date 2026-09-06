# Changelog

## 1.0.3

- The colour-code strip works: `stage.replace` replaces what its capture
  group matches, and 1.0.2's expression had no group.

## 1.0.2

- `job=journal` is set after the source: Alloy stamps its component id on
  journal lines over the source's labels block.
- Terminal colour codes are stripped from Core's lines.
- An icon.

## 1.0.1

- The journal path is chosen at start (`run.sh`) and passed to Alloy
  explicitly. With an empty path Alloy opened the journal "local only" -
  the container's own machine-id - and shipped nothing.

## 1.0.0

- First release. Grafana Alloy v1.19.2, the official image, one static
  config: the host journal (Core, Supervisor, add-ons, the OS) to the
  homelab's VictoriaLogs over the Loki push protocol, with the estate's
  labels (`job`, `host`, `vmid`, `node`, `guest_type`, `unit`, `container`,
  `severity` for the host's units, `stream` for container lines). No options.
