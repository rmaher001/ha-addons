# Richard's homelab add-ons

Home Assistant add-ons that make Home Assistant OS a citizen of the homelab
estate. Each add-on is a directory with its `config.yaml`, `Dockerfile` and
the files it ships; Home Assistant builds the image on the box.

| Add-on | What it does |
|---|---|
| [`alloy`](alloy/) | Ships the host journal - Core, Supervisor, add-ons and the OS - to the estate's log store. |

## Install

Settings → Add-ons → Add-on Store → ⋮ → Repositories → add
`https://github.com/rmaher001/ha-addons`. The add-ons then appear in the
store; install, start. The first build takes a few minutes.

## Update

Every change is a commit here: bump `version` in the add-on's `config.yaml`
and add the entry to its `CHANGELOG.md`; the store shows the update. Nothing
is configured in the UI - an add-on here has no options, so what runs is
what is in git.

## Tests

`pytest tests` pins each add-on's shape (the image pin, the labels, what is
and is not mounted). The behaviour is proven on the box: the add-on's own
log, and the lines arriving in the store.
