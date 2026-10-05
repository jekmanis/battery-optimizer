# Pinned base image. Since Supervisor 2026.04 there is no BUILD_FROM build
# argument: the base is named here, and the date-stamped tag keeps rebuilds
# reproducible. Multi-arch (amd64 + aarch64), s6-overlay + bashio included.
FROM ghcr.io/home-assistant/base-python:3.14-alpine3.24-2026.08.0

# tzdata: the process zone is set to HA's zone at startup (TZ + tzset), so
# the container's libc and zoneinfo need the zone files.
RUN apk add --no-cache tzdata

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# The optimizer - battery_optimizer.py + battery_optimizer_lib/ - is staged
# into app/ by scripts/deploy_addon.py; it is not committed here a second time.
COPY app/ /app/
COPY run.sh /run.sh
RUN chmod a+x /run.sh \
    && python3 -m compileall -q /app \
    && cd /app && python3 -c "import battery_optimizer, battery_optimizer_lib.addon_main"

WORKDIR /app
CMD [ "/run.sh" ]
