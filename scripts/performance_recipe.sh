#!/usr/bin/env bash
# Hardware-aware policy is resolved in Trainer; all values remain overridable.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
COMPILE_WARMUP=${COMPILE_WARMUP:-False}
if [[ "${TRAIN_DTYPE:-float32}" == float32 || "${COMPILE_FRONTEND:-none}" != none ]]; then
    # The external FA3 extension on the target vendor stack has no FakeTensor
    # implementation. Compiled frontends use native SDPA; explicit overrides
    # remain available for environments with a compile-compatible extension.
    USE_FLASH_ATTN3=${USE_FLASH_ATTN3:-False}
fi
PERFORMANCE_ARGS=(--attention_backend "$ATTENTION_BACKEND" --compile_warmup "$COMPILE_WARMUP")
