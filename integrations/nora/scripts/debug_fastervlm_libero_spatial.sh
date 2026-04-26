#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export NORA_COMPARE_METHOD=fastervlm
export NORA_COMPARE_METHOD_TAG=fastervlm
export NORA_DEBUG_PORT=${NORA_DEBUG_PORT:-5679}
exec "$SCRIPT_DIR/debug_compare_libero_spatial_common.sh"
