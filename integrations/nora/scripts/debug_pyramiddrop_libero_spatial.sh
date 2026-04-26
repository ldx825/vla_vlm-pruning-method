#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export NORA_COMPARE_METHOD=pyramiddrop
export NORA_COMPARE_METHOD_TAG=pyramiddrop
export NORA_DEBUG_PORT=${NORA_DEBUG_PORT:-5680}
exec "$SCRIPT_DIR/debug_compare_libero_spatial_common.sh"
