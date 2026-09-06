# Grafana's official image, pinned to one release: the binary, its entrypoint
# (/bin/alloy) and nothing else. To update, bump this tag and the version in
# config.yaml, and say so in CHANGELOG.md.
FROM grafana/alloy:v1.19.2

COPY config.alloy /etc/alloy/config.alloy
COPY run.sh /run.sh

ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_VERSION
LABEL \
    io.hass.name="Alloy (homelab)" \
    io.hass.description="Ships the Home Assistant OS journal to the homelab's log store" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}"

# run.sh picks the journal directory and execs /bin/alloy with its flags.
ENTRYPOINT ["/run.sh"]
