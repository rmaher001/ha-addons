# Python plus a pinned scikit-learn. To update, bump the pins and the version in
# config.yaml, and say so in CHANGELOG.md.
FROM python:3.13-slim

RUN pip install --no-cache-dir numpy==2.5.5 scikit-learn==1.9.1

COPY learner /app/learner
WORKDIR /app

ARG BUILD_ARCH
ARG BUILD_VERSION
LABEL \
    io.hass.name="Water log learner" \
    io.hass.description="Learns Richard's water-use labels" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}"

CMD ["python", "-u", "-m", "learner.main"]
