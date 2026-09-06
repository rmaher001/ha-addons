# Grafana's official image, pinned to one release: the binary, its entrypoint
# (/bin/alloy) and nothing else. To update, bump this tag and the version in
# config.yaml, and say so in CHANGELOG.md.
FROM grafana/alloy:v1.19.2

COPY config.alloy /etc/alloy/config.alloy

ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_VERSION
LABEL \
    io.hass.name="Alloy (homelab)" \
    io.hass.description="Ships the Home Assistant OS journal to the homelab's log store" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}"

# --disable-reporting: no usage report to Grafana. /data is the add-on's own
# persistent directory - the WAL and the journal position live there, so a
# restart resumes where it stopped instead of re-shipping. The write-ahead
# log is an experimental feature in this Alloy release: without
# --stability.level=experimental Alloy refuses the config outright.
CMD ["run", "--disable-reporting", "--stability.level=experimental", "--server.http.listen-addr=0.0.0.0:12345", "--storage.path=/data/alloy", "/etc/alloy/config.alloy"]
