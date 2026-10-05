#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
# with-contenv: SUPERVISOR_TOKEN reaches the process through s6's container env.
cd /app || exit 1
exec python3 -u -m battery_optimizer_lib.addon_main
