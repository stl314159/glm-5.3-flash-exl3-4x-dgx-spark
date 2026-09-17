# Why `EXL3_FAT_KERNEL=0` and `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=0` are the defaults

Fleet: 4x NVIDIA DGX Spark (GB10), switched RoCE, dual HCA, TP=4, MiaAI runtime root at
f906ee9 with the sparse-attention slice patch, `MAX_NUM_SEQS=8`, mixed prefill on.
Workload: a coding-agent harness running four concurrent sessions of 200K-450K tokens each,
with tool calls, stream cancellations and long generations. Dates are 2026-09-13 to 15.

## The symptom

For hours at a time the fleet decoded at 3-12 tok/s aggregate instead of the 70-90 tok/s it
delivers otherwise, with two to four sessions active. Steps took 350-1000 ms instead of 67 ms.
`/health` stayed 200, nothing queued, no errors, KV bytes were never above 20% used, GPUs held
2.18 GHz. From the outside it looked like the agent was over-scheduling.

## What it was not

* Not concurrency: two sessions were as slow as four during an episode, and four sessions ran
  at 85-90 tok/s outside one.
* Not cold prefill interference: samples with no prefill running were the slowest.
* Not memory, swap, compaction or GPU clocks: all measured flat.
* Not the E2 fat-expert kernel: `EXL3_FAT_KERNEL=0` (the workaround from MiaAI issue #143)
  changed nothing; the episode recurred within five hours on the legacy tier.
* Not the runtime root: c190db1 and f906ee9 behaved identically.

## What it was

The KV cache is addressed through one pool of block ids shared by all seven hybrid cache
groups (full attention, four Mamba-style layers, the indexer tail, the DFlash2 drafter's
sliding window). Pages are 3,584 tokens for every group, and the drafter's 64-token blocks are
slot-shared onto the same ids, so one 64-token drafter block costs as much id space as 3,584
tokens of attention. In the default "dense" retention mode the drafter hashes 33 of the ~38
ids a cached 3,584-token segment costs, for a group that the hybrid hit calculation ignores.
MiaAI's `docs/DESIGN-apc-per-group-retention.md` works this out; their two-Spark pool is 642
ids, ours at TP4 is about 1,950 (4.7x concurrency at 1M tokens / 414 ids per max-length request).

With one or two long sessions the contexts stay resident and turns hit the cache (94-98% hit
rate). With three or four, dense-mode id consumption exceeds the pool, each turn evicts another
session's context, and the next turn of that session silently re-prefills its whole 200K+
context at ~1,000 tok/s while every other stream crawls. Evidence during an episode: KV usage
climbing ~830 tokens/s with under 10 tok/s generated, the prefix-cache hit rate at 0-14%, py-spy
showing the chunked prefill kernels in the TP worker, and the drafted-token rate far below what
the running requests would produce if they were decoding. The engine's "prompt throughput"
counter does not count this recompute, which is why it read 0.0 throughout.

## The fix

`patch_apc_per_group_retention` (in MiaAI roots >= f906ee9, baked into the image) makes the
retention interval per group. `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=0` puts the drafter
group on boundary-only retention (~5 ids per cached segment instead of ~38) and leaves the
attention and Mamba groups on the fine hit grid. The launcher never forwarded the variable, so
`tp4-cluster.sh` and `node-launch.sh` now pass it through, and the example config defaults it.
Result on this fleet after the restart's own re-prefill wave: four sessions decoding at 70-96
tok/s aggregate, 34 ms steps at three running, hit rate climbing through the 60s within the
first hour. Per MiaAI's capacity formula the resident budget rises from roughly 50K tokens of
conversation to about 1.4M across all sessions at TP4; sessions much beyond ~350K each will
outgrow it again, and the remaining fix for that is a larger pool (MiaAI issue #122).

Fallback if the per-group setting misbehaves: the global
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=14336` (MiaAI's TP2 default, ~150K tokens of capacity,
coarser hits everywhere). MiaAI's own TP4 launcher refuses the global knob without stating why;
the per-group one is untested by them at TP4 and is validated here only by this fleet's soak.

## `EXL3_FAT_KERNEL=0`

The E2 fat-expert prefill kernel was disabled while it was the suspect. Measured with it off:
single-stream decode 42.6 tok/s (35.6 with it on), cold prefill 1,012-1,142 tok/s (unchanged),
and a smoother incumbent stream during a peer's prefill (max gap 1.1 s vs 6 s). No reason to
turn it back on for this fleet; `1` restores the upstream behavior.

## Deployment log

- **2026-09-15** f906ee9 image, these defaults: four sessions at 200K to 450K tokens decoding at 70 to 96 tok/s
  aggregate after warm-up, 34 ms steps.
- **2026-09-16** rebuilt from MiaAI main 8f29c6d with `recipe/build-image.sh` (same tunables; picks up the KV
  capacity boot log, per-request no-store, InstantTensor, and the decode-floor restart fix; mixed prefill stays `off`).
  Verified: slice patch PASS, retention line `[None,None,None,None,None,None,0]`, single-stream 41.4 tok/s at 68 ms
  steps vs 42.6 on f906ee9. The capacity log corrected the arithmetic above: TP4 pages are 1,792 tokens, the pool is
  3,185 ids, and a cached segment costs 33 ids dense (28 of them the drafter) or 5 with the drafter at boundary-only
  retention, about 1.14M tokens of cached conversation across all sessions.
- **2026-09-17** stack stopped to run the DeepSeek-V4.1-Flash TP4 recipe on the same four nodes; config unchanged.
