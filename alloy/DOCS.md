# Alloy (homelab)

Ships Home Assistant OS's journal to the homelab's log store, VictoriaLogs on
lxc-222, over the Loki push protocol. It is the same Grafana Alloy every
other host in the estate runs; this add-on exists because HAOS is a read-only
appliance that cannot run it as a package.

## What it collects

`journald: true` mounts the host's journal read-only. On HAOS that one
journal holds everything: Home Assistant Core, the Supervisor, every add-on,
docker and the OS itself. On first start the last 12 hours are shipped; after
that the add-on keeps its position in `/data` and a restart resumes.

## Labels

| Label | Value |
|---|---|
| `job` | `journal` |
| `host` | `homeassistant` |
| `vmid` / `node` / `guest_type` | `200` / `proxmox2` / `vm` - the estate's guest identity |
| `unit` | the systemd unit, or the container name for a container's line (`homeassistant` for Core, `hassio_supervisor`, `app_<hash>_<slug>` for an add-on) |
| `container` | the container name, container lines only |
| `severity` | journald priority, the host's own units only |
| `stream` | `stdout` / `stderr`, container lines only - which stream docker read, never the line's level; Core logs to stderr |

A container's real level is in its text (`ERROR (MainThread) [...]`); rules
match on that.

Queries:

```
host:homeassistant unit:homeassistant "Error while executing automation"
host:homeassistant unit:hassio_supervisor
host:homeassistant severity:(err OR crit)          # the host's own units
```

## No options

The store's address and every label are in `config.alloy`, in git. Changing
them is a commit and a version bump, never a form in the UI.

## Updating Alloy

Bump the `FROM grafana/alloy:vX.Y.Z` tag in `Dockerfile`, the `version` in
`config.yaml`, add the changelog entry, push. Home Assistant shows the update
and rebuilds the image on the box.

## Debug UI

Alloy listens on 12345 inside the add-on's network for the Supervisor's
watchdog (`/-/ready`). The port is not exposed to the LAN; it can be, from the
add-on's Network tab, when a look at the pipeline is needed.
