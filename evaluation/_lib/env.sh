# Shared environment setup for evaluation launchers. Meant to be SOURCED,
# not executed: it cds to the repo root and puts it on PYTHONPATH so that
# `wam` / `wam_client` / `evaluation.*` imports work from any CWD.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../_lib/env.sh"

_WAM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_WAM_LIB_DIR}/../.." && pwd)"
export REPO_ROOT

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export HF_HOME=${HF_HOME:-/tmp/abot_m05_hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/abot_m05_xdg_cache}
