# Changelog

## 1.0.0

- First release. Grafana Alloy v1.19.2, the official image, one static
  config: the host journal (Core, Supervisor, add-ons, the OS) to the
  homelab's VictoriaLogs over the Loki push protocol, with the estate's
  labels (`job`, `host`, `vmid`, `node`, `guest_type`, `unit`, `container`,
  `severity` for the host's units, `stream` for container lines). No options.
