#!/bin/bash
# GLM-5.3-Flash EXL3 TP4 — node-local rank launcher (runs on each rank, no SSH inside).
# Required env: NODE_RANK HEAD_IP FABRIC_IP RDV PORT IMG ROOT HF VC TAG DK SPEC_METHOD DRAFT_TP
# Optional env: MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS KV_CACHE_DTYPE DFLASH_TOKENS NCCL_DEBUG EXTRA_ARGS
#               TP_SIZE NNODES (default 4/4; 2/2 requires slicing disabled and is not this qualified recipe)
# EXL3_FAT_KERNEL (0/1): E2 fat-expert prefill kernel of upstream PR77 (images built from c190db1 or later; ignored by 493cb88)
# SWAPPINESS (0-100): set vm.swappiness on this rank before launch (runtime only; persist it yourself in /etc/sysctl.d)
# MIXED_PREFILL_CHUNK (skip default = never mix a new prefill with running decodes; N = cap mixed chunk; off = stock chunked prefill)
set -euo pipefail
for v in NODE_RANK HEAD_IP FABRIC_IP RDV PORT IMG ROOT HF VC TAG DK SPEC_METHOD DRAFT_TP; do [ -n "${!v-}" ] || { echo "missing $v" >&2; exit 2; }; done
slice_env=()
case "${VLLM_SM120_SPARSE_MLA_SLICE_TOKENS:-0}" in
  0) ;;
  64)
    [ "${TP_SIZE:-4}" = 4 ] || { echo 'attention slice=64 requires TP_SIZE=4 (H16); TP2 slicing is unsupported. Use a separately qualified TP2 recipe.' >&2; exit 2; }
    [ -n "${SPARSE_MLA_PATCH_B64:-}" ] || { echo 'missing pinned attention patch' >&2; exit 2; }
    slice_env=(-e VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64 -e SPARSE_MLA_PATCH_B64="$SPARSE_MLA_PATCH_B64")
    ;;
  *) echo 'attention slice must be 0 or 64' >&2; exit 2;;
esac
NETDEV=${NETDEV:-enp1s0f1np1}; HCA=${HCA:-rocep1s0f1}
MODEL_REV=25a44fdbf16862a46b7cc9921142c6c81350af2f
DFLASH_REV=dc77ff1c99eeb2df044ee3d4f0094eb033fee410
MODEL_DIR=/root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/$MODEL_REV
DFLASH_DIR=/root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/$DFLASH_REV
[ -f "$HF/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/$MODEL_REV/config.json" ] || { echo "model snapshot missing" >&2; exit 3; }
[ "$SPEC_METHOD" != dflash ] || [ -f "$HF/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/$DFLASH_REV/model.safetensors" ] || { echo "dflash snapshot missing" >&2; exit 3; }
$DK image inspect "$IMG" >/dev/null 2>&1 || { echo "image $IMG missing" >&2; exit 3; }
ip -4 -o addr show dev "$NETDEV" | grep -q " $FABRIC_IP/" || { echo "$FABRIC_IP not on $NETDEV" >&2; exit 3; }
# Resolve the RoCEv2 GID index that carries FABRIC_IP on this HCA (indices differ per node).
want=$(python3 -c "import ipaddress;ip=int(ipaddress.ip_address('$FABRIC_IP'));print('0000:0000:0000:0000:0000:ffff:%04x:%04x'%(ip>>16,ip&0xffff))")
GID=""
for i in $(seq 0 31); do
  f=/sys/class/infiniband/$HCA/ports/1/gids/$i; [ -f "$f" ] || continue
  [ "$(cat "$f")" = "$want" ] || continue
  grep -q "RoCE v2" "/sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i" 2>/dev/null && { GID=$i; break; }
done
[ -n "$GID" ] || { echo "no RoCEv2 GID for $FABRIC_IP on $HCA" >&2; exit 3; }
LINK=$(ethtool "$NETDEV" 2>/dev/null | awk -F: '/Speed/{gsub(/ /,"",$2);print $2}')
echo "rank=$NODE_RANK host=$(hostname) fabric=$FABRIC_IP gid=$GID link=$LINK"

# NCCL >=2.21 can select each HCA's current RoCEv2 GID by address. A saved
# numeric index can become invalid after a link flap even when the IP survives.
rdma_env=(-e NCCL_IB_HCA="${NCCL_IB_HCA:-$HCA}" -e NCCL_IB_ADDR_FAMILY=AF_INET)
case "${RDMA_GID_MODE:-auto}" in
  auto)
    cidr=${NCCL_IB_ADDR_RANGE:-$(ip -j -4 addr show dev "$NETDEV" | python3 -c '
import ipaddress, json, sys
wanted = sys.argv[1]
addresses = [a for interface in json.load(sys.stdin) for a in interface["addr_info"] if a["local"] == wanted]
assert len(addresses) == 1, "fabric IPv4 address must be unique on the selected interface"
print(ipaddress.ip_network(wanted + "/" + str(addresses[0]["prefixlen"]), strict=False))' "$FABRIC_IP")}
    # Controller embeds the small validator when it streams this launcher.
    [ -n "${RDMA_PREFLIGHT_B64:-}" ] || { echo 'missing all-HCA preflight validator' >&2; exit 3; }
    printf '%s' "$RDMA_PREFLIGHT_B64" | base64 -d | python3 - "${NCCL_IB_HCA:-$HCA}" "$cidr"
    rdma_env+=(-e NCCL_IB_ADDR_RANGE="$cidr") ;;
  fixed)
    [[ "${NCCL_IB_HCA:-$HCA}" != *,* ]] || { echo "fixed GID mode requires one HCA" >&2; exit 3; }
    rdma_env+=(-e NCCL_IB_GID_INDEX="$GID") ;;
  *) echo "RDMA_GID_MODE must be auto or fixed" >&2; exit 3 ;;
esac
[ -z "${NCCL_IB_TC:-}" ] || rdma_env+=(-e NCCL_IB_TC="$NCCL_IB_TC")
[ -z "${NCCL_IB_FIFO_TC:-}" ] || rdma_env+=(-e NCCL_IB_FIFO_TC="$NCCL_IB_FIFO_TC")

# GB10 unified memory: vLLM's startup check counts page cache as used, so reclaim it before the container starts.
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "warn: could not drop caches" >&2
if [ -n "${SWAPPINESS-}" ]; then sudo -n sysctl -q -w "vm.swappiness=$SWAPPINESS" 2>/dev/null || echo "warn: could not set swappiness" >&2; fi
echo "swappiness=$(cat /proc/sys/vm/swappiness) swap-used-MiB=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{printf "%d",(t-f)/1024}' /proc/meminfo)"
echo "mem-available-GiB=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo) mem-free-GiB=$(awk '/MemFree/{printf "%.1f",$2/1048576}' /proc/meminfo)"
STATE=$HOME/AI/tp4-$TAG; mkdir -p "$STATE" "$VC/triton" "$VC/tilelang"
NAME=glm53-tp4-$TAG-r$NODE_RANK
if [ -n "$($DK ps -aq --filter "name=^/$NAME$")" ]; then echo "container $NAME already exists" >&2; exit 4; fi

INNER=$STATE/inner-r$NODE_RANK.sh
cat >"$INNER" <<'INNER'
#!/bin/bash
set -euo pipefail
args=(
  --served-model-name GLM-5.3-Flash-EXL3
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "${TP_SIZE:-4}" --nnodes "${NNODES:-4}" --node-rank "$NODE_RANK"
  --master-addr "$HEAD_IP" --master-port "$RDV"
  --distributed-executor-backend mp
  --tool-call-parser glm47 --enable-auto-tool-choice
  --reasoning-parser glm45 --enable-prefix-caching
  --no-enable-flashinfer-autotune --quantization exl3
  --max-model-len "${MAX_MODEL_LEN:-1000000}" --gpu-memory-utilization "${GPU_MEM_UTIL:-0.87}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --limit-mm-per-prompt '{"image":4,"video":1}' --skip-mm-profiling
  --chat-template /opt/glm53/chat_template.jinja
)
[ "$NODE_RANK" = 0 ] || args+=(--headless)
if [ "$SPEC_METHOD" = dflash ]; then
  args+=(--speculative-config "$(python3 -S -c 'import json,os; print(json.dumps({"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":int(os.environ["DRAFT_TP"])},separators=(",",":")))')")
fi
if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a extra <<<"$EXTRA_ARGS"; args+=("${extra[@]}"); fi
for patch in patch_glm_video_placeholders.py patch_suppress_stops_in_reasoning.py patch_scheduler_decode_floor.py \
  patch_glm5_drafter_group.py patch_hybrid_prefix_hit.py patch_xgrammar_termination.py patch_kpool_tail_slotmap.py patch_ablit.py; do
  [ -f "/opt/glm53/$patch" ] || { echo "required patch absent: $patch" >&2; exit 1; }
  python3 "/opt/glm53/$patch"
done
if [ "${VLLM_SM120_SPARSE_MLA_SLICE_TOKENS:-0}" = 64 ]; then
  printf '%s' "${SPARSE_MLA_PATCH_B64:?}" | base64 -d | python3
fi
echo "[tp4-r$NODE_RANK] launching: vllm serve $MODEL_DIR ${args[*]}"
exec vllm serve "$MODEL_DIR" "${args[@]}"
INNER
chmod 700 "$INNER"

mounts=(
  -v "$HF:/root/.cache/huggingface:ro"
  -v "$VC:/root/.cache/vllm" -v "$VC/triton:/root/.triton/cache" -v "$VC/tilelang:/root/.tilelang/cache"
  -v "$INNER:/start.sh:ro"
  -v "$ROOT/files/chat_template.jinja:/opt/glm53/chat_template.jinja:ro"
)
for p in patch_glm_video_placeholders patch_suppress_stops_in_reasoning patch_scheduler_decode_floor patch_glm5_drafter_group \
  patch_hybrid_prefix_hit patch_xgrammar_termination patch_kpool_tail_slotmap ablit_runtime patch_ablit; do
  [ -f "$ROOT/overlay/$p.py" ] || { echo "overlay $p.py missing in $ROOT" >&2; exit 3; }
  mounts+=(-v "$ROOT/overlay/$p.py:/opt/glm53/$p.py:ro")
done

$DK run -d --name "$NAME" \
  --label glm53.tp4.tag="$TAG" --label glm53.tp4.rank="$NODE_RANK" \
  --restart=no --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
  "${mounts[@]}" \
  -e NODE_RANK="$NODE_RANK" -e HEAD_IP="$HEAD_IP" -e RDV="$RDV" -e PORT="$PORT" \
  -e MODEL_DIR="$MODEL_DIR" -e DFLASH_MODEL_DIR="$DFLASH_DIR" -e SPEC_METHOD="$SPEC_METHOD" -e DRAFT_TP="$DRAFT_TP" \
  -e TP_SIZE="${TP_SIZE:-4}" -e NNODES="${NNODES:-4}" \
  -e DFLASH_TOKENS="${DFLASH_TOKENS:-7}" -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}" -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}" -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}" \
  ${VLLM_USE_BREAKABLE_CUDAGRAPH:+-e VLLM_USE_BREAKABLE_CUDAGRAPH="$VLLM_USE_BREAKABLE_CUDAGRAPH"} \
  ${VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA:+-e VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA="$VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"} \
  -e EXTRA_ARGS="${EXTRA_ARGS:-}" -e EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-0}" -e HOST_SWAPPINESS="$(cat /proc/sys/vm/swappiness)" \
  -e NCCL_SOCKET_IFNAME="$NETDEV" -e GLOO_SOCKET_IFNAME="$NETDEV" -e VLLM_HOST_IP="$FABRIC_IP" \
  "${rdma_env[@]}" "${slice_env[@]}" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  ${NCCL_PROTO:+-e NCCL_PROTO="$NCCL_PROTO"} ${NCCL_ALGO:+-e NCCL_ALGO="$NCCL_ALGO"} ${NCCL_IB_TIMEOUT:+-e NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT"} \
  ${NCCL_IB_RETRY_CNT:+-e NCCL_IB_RETRY_CNT="$NCCL_IB_RETRY_CNT"} ${NCCL_BUFFSIZE:+-e NCCL_BUFFSIZE="$NCCL_BUFFSIZE"} \
  ${NCCL_MIN_NCHANNELS:+-e NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS"} ${NCCL_MAX_NCHANNELS:+-e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS"} \
  ${NCCL_IB_QPS_PER_CONNECTION:+-e NCCL_IB_QPS_PER_CONNECTION="$NCCL_IB_QPS_PER_CONNECTION"} ${NCCL_IB_SPLIT_DATA_ON_QPS:+-e NCCL_IB_SPLIT_DATA_ON_QPS="$NCCL_IB_SPLIT_DATA_ON_QPS"} \
  ${NCCL_NET_GDR_LEVEL:+-e NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL"} \
  -e HF_HOME=/root/.cache/huggingface -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e GLM53_SUPPRESS_STOPS_IN_REASONING=1 -e GLM53_MIXED_PREFILL_CHUNK="${MIXED_PREFILL_CHUNK:-skip}" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e ABLIT=0 \
  --entrypoint bash "$IMG" /start.sh >/dev/null
echo "started $NAME"
