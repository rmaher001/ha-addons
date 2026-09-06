#!/bin/sh
# The only code of ours that runs: pick the directory that holds the host's
# journal, then hand over to Alloy.
#
# The Supervisor mounts the host's journal read-only. Where HAOS keeps it
# depends on the host (/var/log/journal when persistent, /run/log/journal
# when volatile), and the path must be EXPLICIT: an empty `path` makes Alloy
# open the journal "local only", which inside a container means the
# container's own machine-id - it opens fine and reads nothing (found live,
# 2026-09-05: zero lines shipped).
set -eu
JOURNAL_PATH=/var/log/journal
if [ -z "$(ls -A "$JOURNAL_PATH" 2>/dev/null)" ]; then
    JOURNAL_PATH=/run/log/journal
fi
echo "alloy (homelab): reading the host journal at ${JOURNAL_PATH}"
export JOURNAL_PATH
# --disable-reporting: no usage report to Grafana. --stability.level: the
# write-ahead log is an experimental feature in this Alloy release; without
# the flag Alloy refuses the config. /data is the add-on's own persistent
# directory - the WAL and the journal position live there, so a restart
# resumes where it stopped instead of re-shipping.
exec /bin/alloy run --disable-reporting --stability.level=experimental --server.http.listen-addr=0.0.0.0:12345 --storage.path=/data/alloy /etc/alloy/config.alloy
