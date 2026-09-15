# GLM-5.3-Flash EXL3 TP4 sparse-attention recipe

**Build provenance:** qualification used local per-rank image builds. The upstream
GHCR `:exl3` image is not the artifact we qualified, and equivalence has not been
established. This repository does not yet provide a pullable qualified image or
a verified reproducible build. See [recorded image IDs, overlay hashes and
reproduction details](REQUIREMENTS.md#recorded-native1024-build-identities).

This recipe selects one four-rank GB10 configuration: EXL3
with FP8 KV cache, DFlash2 at three speculative tokens, eager execution, a
64-row sparse-MLA attention slice, mixed prefill `off`, a 2,048-token outer
batch, and native long-prefill threshold 1024. It launches a compatible, pinned upstream
image and runtime root; it is not a standalone vLLM distribution.

Native1024 with eight sequences completed one bounded 150-minute mixed-load qualification and was deployed on 7 September after a successful deliberate rollback drill. A separately qualified 16-sequence batch setting is documented below; it trades interactive latency for capacity and is not the default.
It is not fresh-install, reboot, indefinite-stability, maximum-context, or
global-optimum evidence. See [EVIDENCE.md](EVIDENCE.md) and the
[historical H16 baseline projection](qualification-attention64h16.json).

Older files under `evidence/` and the historical experiment queues describe
earlier configurations. Their successful short runs are not pooled into this
configuration's qualification; use the selected settings and commands below.

## Selected result

| Measurement | Native1024 full qualification |
|---|---:|
| Accounted HTTP requests / errors | 484 / 0 |
| Mixed soak duration; requests; exact retrieval | 151.9 min; 368; 92 / 92 |
| Coding aggregate completion | 62.89 tok/s |
| Natural 4,096-token long-generation decode | 33.56 tok/s |
| Cold prefill, 282,310-token prompt | 1,010.5 tok/s |
| Peak concurrent-8 aggregate completion | 126.61 tok/s |
| Mixed-soak aggregate completion | 34.43 tok/s |
| Mixed short-request wall / first-token p95 | 5.511 s / 3.108 s |

`qualification-attention64h16.json` remains the historical H16/off baseline
projection. The native result is bounded evidence for this workload and revision,
not fresh-install, reboot, indefinite-stability, maximum-context, or global-optimum evidence.

## September 9 RigMark challenger screen

A pinned short RigMark comparison found faster coding and cold prefill with a
separate NVFP4/Marlin recipe, alongside replay-latency and first-use concurrency
trade-offs. The EXL3 recipe above remains the qualified default.

| RigMark metric (median) | EXL3 TP4 | NVFP4 TP4 | NVFP4 change |
|---|---:|---:|---:|
| Code decode | 49.6 tok/s | 68.0 tok/s | +37.1% |
| Prose decode | 28.6 tok/s | 28.4 tok/s | -0.6% |
| Structured decode | 55.9 tok/s | 91.3 tok/s | +63.3% |
| 64k cold prefill | 1,087 tok/s | 1,393 tok/s | +28.2% |
| 64k replay TTFT | 1.158 s | 2.353 s | +103.2% (slower) |
| C4 aggregate | 93.1 tok/s | 81.3 tok/s | -12.6% |

Both passed 9/9 basic decode-output checks and 35/35 expected benchmark POSTs.
NVFP4's two C4 rounds were 62.9 and 99.7 tok/s; the first overlaps an inference-time
compilation warning. Both rounds remain in the result. This is a whole-recipe
comparison, with three decode samples per category, one prefill/replay pair per
depth and two concurrency rounds. It is not semantic coding-quality evidence,
a new soak, or directly comparable with the qualification table above.

A separate JSpark3-inspired 2 ms queue-spin trial showed no material decode gain
and lower C4 throughput; it was rejected. A TP4 KDA projection prototype remains
microbenchmark-only. See [the full protocol, source pins and evidence](EVIDENCE.md#september-9-matched-rigmark-screening).

## September 7 bounded capacity and transport screens

The selected native1024 eight-sequence recipe remains the qualified default. The following later measurements use the same model/backend lane and frozen workload, but do not replace its qualification.

| Screen | Matched observation | Status |
|---|---|---|
| `MAX_NUM_SEQS=16`, initial capacity screen | C16 aggregate completion `+36.26%` | Screen only |
| `MAX_NUM_SEQS=16`, forward mixed pair | C16 `187.52` vs `133.18` tok/s (`+40.80%`) | Default guard failed: short-completion p95 `+49.13%` |
| `MAX_NUM_SEQS=16`, reverse mixed pair | C16 `188.19` vs `132.19` tok/s (`+42.36%`) | Capacity evidence only; does not repair the forward default-guard failure |
| Native prefill threshold 1536 | Cold prefill `+3.23%` | Rejected: below the predeclared 5% target |
| Native prefill threshold 2048 | Cold prefill `+4.39%` | Rejected: below the predeclared 5% target |
| DFlash draft length 4 | Declared target did not reach 5% in its fresh-control screen | Rejected |
| Outer batch 4096 | Mixed aggregate output `+0.42%` | Rejected: below the predeclared 5% target |
| QPS4 plus split-data transport | Long generation `28.87` vs `31.82` tok/s (`-9.27%`) | Rejected; short screen only, **NO SOAK** |

The capacity rows describe aggregate work at offered concurrency 16. They are not a claim that a single interactive stream is 40% faster. The forward pair fails the interactive-default guard, so native1024 remains selected. The QPS candidate changed only `NCCL_IB_QPS_PER_CONNECTION=4` and `NCCL_IB_SPLIT_DATA_ON_QPS=1`; its observed decode difference is not causal attribution to either setting. The recipe already uses both HCAs, cross-NIC routing and four channels. [NVIDIA describes the two PCIe paths](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html); NCCL documents [QPs per connection](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-qps-per-connection) and [split-data behavior](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-split-data-on-qps). A supplied [two-Spark result](https://x.com/Khen_na_/status/2096709963263418515) motivated this fresh test; its DeepSeek/direct-cable result is not a local GLM TP4 measurement.

### Qualified optional batch capacity

The fresh r2 protocol and independent review passed: 16 actual workers, all frozen gates, unchanged four-rank source and identity across 34 snapshots, complete HTTP accounting, exact native restoration, real gateway inference, and the signed runtime gate. Native1024 with `MAX_NUM_SEQS=8` remains the serving default; the 16-sequence setting is an opt-in batch/capacity choice.

| Full r2 observation | Result |
|---|---:|
| Varied soak duration / requests | 153.1 min / 493 |
| Accounted HTTP 200 / non-200 / foreign requests | 634 / 0 / 0 |
| Exact long-prompt retrieval | 120 / 120 |
| Minimum available memory across ranks | 17.70 GiB |
| Coding aggregate completion | 59.33 tok/s |
| Natural 4,096-token long decode | 30.42 tok/s |
| Cold prefill, 281,750 prompt tokens | 1,010.8 tok/s |
| C4 / C8 / C12 / C16 aggregate completion | 89.20 / 130.14 / 164.14 / 181.37 tok/s |
| Mixed aggregate completion / prompt throughput | 46.70 / 782.90 tok/s |
| Mixed short completion / first-token p95, n=185 | 181.912 / 169.948 s |
| Mixed long-generation decode p50 | 3.42 tok/s |

The 16-worker mixed saturation has unacceptable latency for the interactive default. Its 46.70 tok/s mixed output is not a matched improvement over the native default's 34.43 tok/s result: that qualification offered four workers. The earlier matched capacity pairs used four mixed workers plus fixed C16 bursts. Their C16 median per-request decode fell from 17.60 to 12.945 tok/s forward and 17.13 to 12.615 tok/s in reverse, even while aggregate completion increased. More capacity does not mean faster individual decoding.

The first full16 attempt was invalid and aborted: its transient launcher expanded Bash rank arrays before the child shell sourced configuration, leaving empty SSH hosts for memory probes. The launcher was corrected and its real memory transport checked before r2. R1 contributes no qualification evidence and is not pooled with r2. These results are bounded to the pinned stack and workload; they do not establish reboot survival, indefinite reliability, maximum context, or a global optimum.

## September 6 one-knob screens (provisional)

These short screens held the selected H16 eager recipe fixed and changed one
scheduler setting at a time. They used the same frozen runner and hard gates,
in the fixed-budget evaluation style described by [Karpathy's
autoresearch](https://github.com/karpathy/autoresearch). They are not part of
the selected qualification.

| Screen | Coding aggregate tok/s | Long decode tok/s | Cold prefill tok/s | 8-stream aggregate tok/s | Score |
|---|---:|---:|---:|---:|---:|
| Baseline H16, threshold 0 | 58.96 | 29.59 | 1,073.1 | 121.84 | 1.000 |
| Mixed prefill 64 | 54.49 | 32.41 | 1,049.4 | 121.80 | 1.678 |
| Mixed prefill 128 | 56.50 | 29.57 | 1,045.1 | 125.31 | 1.599 |
| Native long-prefill threshold 1024 | 61.56 | 35.12 | 1,032.5 | 125.65 | 1.541 |

The native threshold screen advanced to two matched mixed-tail screens and a
bounded full qualification before the verified live cutover. The score is a weighted
tradeoff, not a throughput result. The 128 screen had a better tiny cold-tail
observation but worse cron latency. Tiny-tail samples were n=3, and none of
these screens proves a global optimum, a promotion, or long-run candidate
reliability. The matched screen requires zero errors and foreign requests, at
least 20 short probes, at least a 20% p95 gain, and no more than a 5% aggregate
throughput loss. Native1024 is the selected mixed-load default; individual generation and first-token results remain variable.

## Matched mixed20 selection evidence

Both orderings met the declared screen: 45 short probes per run, zero workload
errors and foreign requests, at least 20% lower short completion-wall p95, and
no more than 5% aggregate-output loss. In r1, p95 fell 19.573 to 7.030 s
(64.1%) while aggregate output rose 50.70 to 50.77 tok/s (+0.14%); short TTFT
p95 rose 0.491 to 3.211 s. In reverse-order r2, p95 fell 22.562 to 17.331 s
(23.2%) while aggregate output fell 50.81 to 50.19 tok/s (-1.22%); TTFT p95
improved 5.595 to 3.087 s. Long decode moved in opposite directions: +2.94%
in r1 and -18.8% in r2. These paired screens support this mixed completion-tail
objective; they do not show universal latency or throughput improvement.

## What the patch addresses

Earlier unsliced graph and fully eager candidates both stalled. In the decisive
capture, one rank held a single resident
`sparse_mla_prefill_kernel<...,16,2048,64>` block at the same program counter
across three samples while TP peers waited in NCCL. Directional state had
incoming data and peer credit, but no outgoing GPU data publication from that
rank. That localizes the observed progress frontier to the GPU/kernel path; it
does not identify an internal race or prove a general FlashInfer defect.

The patch splits only the final guarded sparse-attention call into rows of at
most 64 tokens, preserving the outer 2,048-token scheduler batch, weights,
drafter, cache layout, fabric and upstream overlays. It accepts one source
SHA-256 and produces one patched SHA-256; it rejects any other backend or
geometry. Independent numerical checks covered 1, 64, 65, 129, 193 and 2,048
token rows, including ragged/empty rows, then the selected configuration
completed the bounded qualification.

An independently pinned report describes a similar H16 prefill-kernel wedge,
but it is not maintainer confirmation of the same cause here. Its workaround
is not public source-equivalent. The public Jasl Triton code is also not a
drop-in replacement: its packed FP8/scales and head handling differ from this
GLM `fp8_ds_mla` cache, so it would need a GLM adapter and new numerical and
all-rank qualification. See [Mark Sunner's pinned evidence](https://github.com/marksunner/glm52-dgx-spark-deadlock-evidence/tree/32133d5ef0e4dde00d1da15d639c8a71c92f75f8)
and [Jasl revision `2dd63d85`](https://github.com/jasl/vllm/tree/2dd63d85f4133cf98721ecf6a8d373e4c1dc356f).

## Pinned inputs and effective runtime

| Input | Pin |
|---|---|
| EXL3 checkpoint | [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) @ `25a44fdbf16862a46b7cc9921142c6c81350af2f` |
| DFlash2 drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Upstream runtime root / overlays | [MiaAI-Lab recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/c190db1ae17ba8dff20129ed1f308d10c63cf37d) @ `c190db1ae17ba8dff20129ed1f308d10c63cf37d` |
| Effective engine build | `vllm-0.1.dev20051+g487ecf187-tp4-d6e0b989` |
| Runtime libraries | FlashInfer 0.6.17; Torch 2.13.0+cu130; CUDA runtime 13.0.96; NCCL 2.30.7; Triton 3.7.1 |
| Driver / CUDA observation | NVIDIA 580.173.02; CUDA driver API 13000; CUDA runtime 13.0.96 |
| Selected chat template | `recipe/chat_template-20260904.jinja` SHA-256 `bdc5009ef6024a700f2ab2b8caefb14d083f504cf8d2ce70caa7e459b01cc331` |
| Sparse MLA patch | `recipe/patch_sparse_mla_slice.py`: source `d665ef…cd01`; patched `f1854c…620c6` |

The selected template is an explicit local replacement, not the unmodified
template from upstream `c190db1`. Qualification used independently built images
on each rank; their recorded overlay/template and patched-backend hashes matched. Use a
compatible image digest on each of your ranks, then verify the hashes at
startup. Do not infer that one image digest was used everywhere in the source
qualification.

For recorded per-rank **local image IDs**, E2 kernel-path evidence, dated
kernel/clock observations, the exact long-generation request and a short
diagnostic protocol, see [reproduction details](REQUIREMENTS.md#recorded-native1024-build-identities).
These records do not provide a pullable qualification image or prove that a
different image with the same service fingerprint is equivalent.

The root model is [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
Read [NOTICE.md](NOTICE.md) before serving: DFlash2 makes the selected stack
non-commercial under its published terms.

## Fleet defaults changed in this fork

`cluster.env.example` now defaults `EXL3_FAT_KERNEL=0` and
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=0`, and the launcher forwards the latter to the ranks.
The symptom, the block-id-pool mechanism and the measurements are in [DEFAULTS.md](DEFAULTS.md).

## Prepare, configure and launch

Follow [REQUIREMENTS.md](REQUIREMENTS.md). Build or obtain the upstream image
at the pinned revision, retain its matching `overlay/`, and cache the pinned
model snapshots on each rank. Configure a site-specific file from the exported
repository root:

```bash
cp cluster.env.example cluster.env
$EDITOR cluster.env
CFG="$(pwd)/cluster.env"
source "$CFG"
```

Install the selected template into each rank's configured runtime root before
launch. Repeat this step on every rank, using that rank's actual `ROOT`:

```bash
install -m 0644 recipe/chat_template-20260904.jinja "$ROOT/files/chat_template.jinja"
shasum -a 256 "$ROOT/files/chat_template.jinja"
```

On the controller, from the exported repository root:

```bash
./recipe/tp4-cluster.sh "$CFG" preflight
./recipe/tp4-cluster.sh "$CFG" launch
./recipe/tp4-cluster.sh "$CFG" wait
./recipe/tp4-cluster.sh "$CFG" ready
python3 recipe/functional.py http://YOUR_HEAD_IP:8890/v1
CFG="$CFG" SOAK_MIN=150 SOAK_WORKERS=4 LONGGEN_TOKENS=4096 \
  SOAK_LONGGEN_TOKENS=4096 COLD_TOKENS=280000 CONC_LEVELS=4,8 \
  SOAK_KINDS=short,coding,medium_gen,long_prompt,short,long_gen,long_prompt_96k,short \
  ./recipe/workload-run.sh attention64h16 \
  warmup,sanity,coding,longgen,cold,conc,cancel,soak,sanity_end
```

`tp4-cluster.sh` sources its config before locating its own directory, while
`workload-run.sh` and `tp4-watchdog.sh` change into `recipe/` first. Use an
absolute `CFG` path for all three. `functional.py` posts to
`/chat/completions`, so it requires the `/v1` endpoint root.

`cluster.env.example` enables the selected patch. Its controller streams the
installer into each fresh container. Do not set
`VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64` for another image revision: the patch
rejects a different preimage.
Slicing 64 is required for this selected **TP4/H16** recipe; the node launcher
rejects TP2 with slicing enabled before starting a container. Value 0 is an
unqualified diagnostic path. See [compatibility and startup notes](REQUIREMENTS.md#tp4-slice-compatibility-and-startup).

## Operation and recovery

The exported watchdog checks all rank containers and real inference rather
than trusting `/health` alone. It restarts the retained four-container group,
then requires all-rank readiness. It rate-limits recovery and pauses on a
recovery failure or limit; it does not create an image from changed inputs.

```bash
CFG="$(pwd)/cluster.env"
nohup ./recipe/tp4-watchdog.sh "$CFG" > tp4-watchdog.out 2>&1 &
```

Use `touch "$MAINT"` after sourcing the configuration to pause a configured
watchdog before planned work, and use the same absolute `CFG` with
`./recipe/tp4-cluster.sh "$CFG" stop` to remove a test group. Define an
automatic fallback only after qualifying it independently. The included
`autoresearch_score.py` preserves the frozen receipt scorer used to assess
workload outputs; do not compare results without its hard gates.

Run the complete local suite without GPUs or model access:

```bash
python3 -m unittest discover -s tests -v
```

These 12 tests check scoring rejection and cancellation deadlines. They do
not establish model readiness, numerical equivalence, or sustained reliability.

## Rejected graph mode and fabric boundary

Graph mode is not selected: its full trial completed the frozen model suite,
but had a worse mixed short-request tail and failed its service handback. The
eager choice favors the measured interactive tail; it does not claim that
graphs are generally slower. Coordinated PFC/ECN and dual-HCA/four-channel
configuration improved the fabric baseline, but the surviving captured
progress frontier was in the H16 prefill kernel with clean monitored transport
deltas. That evidence does not support replacing the switch as a remedy.

## Related sources and non-equivalent alternatives

- The original [MiaAI-Lab TP2 recipe at `c190db1`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/c190db1ae17ba8dff20129ed1f308d10c63cf37d) is implementation provenance, not a validated four-rank switched-RoCE replacement.
- [NCCL issue #2353](https://github.com/NVIDIA/nccl/issues/2353) was reviewed and withdrawn as a candidate: this run used 2.30.7 and did not show that issue's required `ncclLocalOpAppend` / sleeping-proxy signature. [NCCL #2334](https://github.com/NVIDIA/nccl/issues/2334) remains only a topology/workload analogy.
- The fabric campaign used RouterOS and switch-marvell 7.23.5 with coordinated PFC/ECN. The vendor [RouterOS changelog](https://mikrotik.com/download/changelogs) documents release provenance; no cable or link-rate change is credited, and fabric changes did not by themselves explain the surviving kernel frontier.
- [tonyd2wild's pinned NVFP4 forensic note](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/blob/8fd2fcd27c04c7fa93e770000b818657f338875d/docs/SM121-CRASH-FORENSICS-2026-08-27.md) retracts an earlier hardware-memory-ceiling diagnosis. Its NVFP4/Marlin stack is not comparable to this EXL3/vLLM result.
- [cfontes' original NVFP4 card](https://huggingface.co/cfontes/glm-5.3-flash-dflash2-tp4/tree/9417f8e107bf373750fd9b1cadaf9a8a70d530d9) and [later drafter card](https://huggingface.co/cfontes/GLM-5.3-Flash-DFlash2-TP4-Spark/tree/fe78d4cebbaebb3744acdde6311df91b3377e2fb) use different weights; the later card reports no served acceptance gain in its rechecks.
- [Pinned SGLang TP4 recipe](https://github.com/joesinvestments/GLM-5.3-Flash-FP8-4x-DGX-Spark/tree/880efbc7793d06a21908afd590d34cd59ca2e00b) is a credible native-FP8 alternative, but differs in engine, weights, speculation, context limit and validation duration. It needs a fresh matched qualification.
- [NVIDIA DGX Spark clustering](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html), [vLLM](https://github.com/vllm-project/vllm), and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) describe the upstream platform components.

## Bounded 64k profiler observation

A healthy 64k-prefill workload captured four host steps per rank over 3.878–4.133 seconds. Within each rank, NCCL all-reduce accounted for 33.3–37.0% of **summed CUDA-kernel durations**, EXL3 MoE/GEMM for 33.9–35.7%, and sparse MLA for 4.8–5.1%. These shares use kernel-duration sums as their denominator, not request wall time or fabric utilization.

This is an early window of a workload that includes overlapping short probes, not a whole-64k or steady-decode profile. GPU annotation mirrors are excluded from host-step counts and spans. Kernels can overlap, and NCCL duration includes peer synchronization and waiting; these measurements do not isolate the switch or wire, or predict an end-to-end speedup.

## Comparable-source boundary

[Tech2Wild's EXL3 recipe](https://github.com/tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark/tree/dc91a125fc60349ce99498d65dac5bc772a43c54) is TP2. Its public TP4 recipe uses a different [NVFP4/Marlin stack](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/tree/8fd2fcd27c04c7fa93e770000b818657f338875d), so neither its headline nor topology belongs in an EXL3/native1024 comparison. [`antirez/ds4`](https://github.com/antirez/ds4/tree/b6af0adf8ca97c89145c9f9c15be70c9fd6c4507) documents a single-GPU DGX Spark GLM path and in-host CUDA multi-GPU mode; its pinned sources do not establish Spark-to-Spark RDMA tensor parallelism. These are implementation references, not local performance evidence.

## Optional batch-capacity setting

To reproduce the qualified batch setting, copy your validated site configuration to `capacity16.env`, change `MAX_NUM_SEQS` to `16`, and give it distinct `TAG` and `MAINT` values. Retain the default configuration. If a group is already serving, enter planned maintenance and stop it using its existing configuration before launching the capacity group. Qualify the result on your own hardware with the full workload:

```bash
CFG="$(pwd)/capacity16.env"  # configured for this site; distinct TAG and MAINT
./recipe/tp4-cluster.sh "$CFG" preflight
./recipe/tp4-cluster.sh "$CFG" launch
./recipe/tp4-cluster.sh "$CFG" wait
CFG="$CFG" SOAK_MIN=150 SOAK_WORKERS=16 LONGGEN_TOKENS=4096 \
  SOAK_LONGGEN_TOKENS=4096 COLD_TOKENS=280000 CONC_LEVELS=4,8,12,16 \
  SOAK_KINDS=short,coding,medium_gen,long_prompt,short,long_gen,long_prompt_96k,short \
  ./recipe/workload-run.sh capacity16 \
  warmup,sanity,coding,longgen,cold,conc,cancel,soak,sanity_end
```

`tp4-cluster.sh` sources the configuration and refuses to launch over an existing same-tag group. `workload-run.sh` sources `CFG` and forwards `SOAK_WORKERS` and `CONC_LEVELS`. The recorded qualification covers our pinned stack; the command reproduces its workload on another site and does not confer qualification on that site.
