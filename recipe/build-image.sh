#!/usr/bin/env bash
# Build the MiaAI GLM-5.3-Flash EXL3 runtime image at a pinned commit, with this
# recipe's chat template installed in the runtime root, then verify it against the
# pieces this launcher depends on and ship image + root to the worker ranks.
#
# usage: recipe/build-image.sh /abs/path/cluster.env <miaai-commit> [build|verify|ship|all]
#
# Reads from cluster.env: RANK_USER, HEAD_IP, RANK_IPS, ROOT, RANK_IMG, SSH_OPTS.
# Optional env: MIAAI_REPO (local clone; default ~/src/github.com/MiaAI-Lab/...),
#               GLM53_RECIPE_STAMP (image label; default <commit>-punkjazz),
#               SHIP_IPS (space-separated worker addresses for the transfer; default the
#               non-head RANK_IPS. Set it to a LAN path when the RoCE fabric is busy serving.)
set -euo pipefail
CFG=${1:?usage: build-image.sh /abs/cluster.env <miaai-commit> [build|verify|ship|all]}
PIN=${2:?MiaAI commit to build}
PHASE=${3:-all}
# shellcheck disable=SC1090
source "$CFG"
HERE=$(cd "$(dirname "$0")" && pwd)
MIAAI_URL=https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git
MIAAI_REPO=${MIAAI_REPO:-$HOME/src/github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}
IMG=${RANK_IMG[0]}
STAMP=${GLM53_RECIPE_STAMP:-${PIN}-punkjazz}
TEMPLATE=$HERE/chat_template-20260904.jinja
# shellcheck disable=SC2206
SSH=(ssh $SSH_OPTS)
log() { printf '%s [build-image] %s\n' "$(date +%T)" "$*"; }

for img in "${RANK_IMG[@]}"; do
  [ "$img" = "$IMG" ] || { echo "RANK_IMG must name one image on every rank (got $img vs $IMG)" >&2; exit 2; }
done

do_build() {
  [ -d "$MIAAI_REPO/.git" ] || git clone "$MIAAI_URL" "$MIAAI_REPO"
  git -C "$MIAAI_REPO" fetch -q origin
  local want; want=$(git -C "$MIAAI_REPO" rev-parse --verify "${PIN}^{commit}")
  if [ ! -e "$ROOT" ]; then
    log "creating runtime root $ROOT at $PIN"
    git -C "$MIAAI_REPO" worktree add --detach "$ROOT" "$want"
  fi
  local have; have=$(git -C "$ROOT" rev-parse HEAD)
  git -C "$ROOT" merge-base --is-ancestor "$want" "$have" \
    || { echo "$ROOT is at $have, which does not contain $PIN ($want); refusing to build" >&2; exit 2; }
  install -m 0644 "$TEMPLATE" "$ROOT/files/chat_template.jinja"
  log "template $(sha256sum "$ROOT/files/chat_template.jinja" | cut -c1-16) installed in $ROOT/files"
  log "docker build $IMG (stamp $STAMP) from $ROOT"
  docker build -t "$IMG" --build-arg GLM53_RECIPE_STAMP="$STAMP" "$ROOT"
}

do_verify() {
  log "verify $IMG"
  local got; got=$(docker image inspect "$IMG" --format '{{index .Config.Labels "glm53.recipe.stamp"}}')
  [ "$got" = "$STAMP" ] || { echo "image stamp is '$got', expected '$STAMP'" >&2; exit 3; }
  # The sparse-MLA slice patch is streamed into each container at start and refuses an
  # unknown preimage. Prove the new image's backend file is the one it expects.
  docker run --rm --entrypoint python3 \
    -v "$HERE/patch_sparse_mla_slice.py:/tmp/slice.py:ro" \
    -e VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64 "$IMG" /tmp/slice.py \
    | tee /dev/stderr | grep -q '"status": "PASS"' || { echo "slice patch preimage check failed" >&2; exit 3; }
  # Overlays the site config relies on must be present in the installed vLLM.
  docker run --rm --entrypoint bash "$IMG" -c '
    set -e; V=/usr/local/lib/python3.12/dist-packages/vllm
    grep -q "glm53-apc-per-group" $V/v1/core/kv_cache_coordinator.py   && echo "  per-group retention overlay: present"
    grep -q "glm53-decode-floor" $V/v1/core/sched/scheduler.py          && echo "  decode-floor/mixed-prefill overlay: $(grep -o "glm53-decode-floor:v[0-9]*" $V/v1/core/sched/scheduler.py | sort -u | tail -1)"
    grep -q "EXL3_FAT_KERNEL" $V/model_executor/layers/quantization/exl3.py && echo "  EXL3_FAT_KERNEL honoured by exl3 overlay"
    python3 -c "import instanttensor" 2>/dev/null && echo "  instanttensor loader: present" || echo "  instanttensor loader: absent"'
  log "verify OK"
}

do_ship() {
  local ips=()
  if [ -n "${SHIP_IPS:-}" ]; then read -r -a ips <<<"$SHIP_IPS"; else
    for ip in "${RANK_IPS[@]}"; do [ "$ip" = "$HEAD_IP" ] || ips+=("$ip"); done; fi
  log "ship $IMG and $ROOT to: ${ips[*]}"
  for ip in "${ips[@]}"; do
    ( docker save "$IMG" | "${SSH[@]}" "$RANK_USER@$ip" docker load >/dev/null \
        && rsync -a --delete --exclude .git -e "ssh $SSH_OPTS" "$ROOT/" "$RANK_USER@$ip:$ROOT/" \
        && log "SHIP_OK $ip" || log "SHIP_FAIL $ip" ) &
  done; wait
  local tsum; tsum=$(sha256sum "$ROOT/files/chat_template.jinja" | cut -c1-16)
  for ip in "${ips[@]}"; do
    "${SSH[@]}" "$RANK_USER@$ip" "s=\$(docker image inspect '$IMG' --format '{{index .Config.Labels \"glm53.recipe.stamp\"}}' 2>/dev/null || echo MISSING); t=\$(sha256sum '$ROOT/files/chat_template.jinja' 2>/dev/null | cut -c1-16); echo \"  \$(hostname): image stamp=\$s template=\$t\"; [ \"\$s\" = '$STAMP' ] && [ \"\$t\" = '$tsum' ]" \
      || { echo "rank $ip does not match the head" >&2; exit 4; }
  done
  log "ship OK"
}

case "$PHASE" in
  build) do_build ;; verify) do_verify ;; ship) do_ship ;;
  all) do_build; do_verify; do_ship ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
