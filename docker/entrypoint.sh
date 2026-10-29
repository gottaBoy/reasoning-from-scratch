#!/bin/sh
set -eu

# A named volume hides /workspace from the image. Seed it once with the
# repository so notebooks and scripts are available without a host bind mount.
if [ ! -f /workspace/README.md ]; then
    cp -a /opt/repo/. /workspace/
fi

if [ -f /workspace/docker/smoke_examples.sh ]; then
    chmod +x /workspace/docker/smoke_examples.sh
fi

exec "$@"
