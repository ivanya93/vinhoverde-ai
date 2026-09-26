#!/bin/bash
# Make the container user match whoever owns the mounted notebooks folder (copied from the
# class stack). Without it, on Linux hosts whose uid is not 1000 you cannot save notebooks.
set -euo pipefail

WORK_DIR="/home/jovyan/work"

if [ -d "${WORK_DIR}" ]; then
    owner_uid="$(stat -c '%u' "${WORK_DIR}")"
    owner_gid="$(stat -c '%g' "${WORK_DIR}")"
    if [ "${owner_uid}" != "0" ]; then
        export NB_UID="${owner_uid}"
        export NB_GID="${owner_gid}"
        export CHOWN_HOME="yes"
        export CHOWN_HOME_OPTS="-R"
        echo "[entrypoint] adopting host ownership of ${WORK_DIR}: uid=${NB_UID} gid=${NB_GID}"
    fi
fi

exec /usr/local/bin/start.sh "$@"
