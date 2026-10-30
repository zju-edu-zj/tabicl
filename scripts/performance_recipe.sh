#!/usr/bin/env bash
# Hardware-aware policy is resolved in Trainer; all values remain overridable.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
COMPILE_WARMUP=${COMPILE_WARMUP:-False}
PERFORMANCE_ARGS=(--attention_backend "$ATTENTION_BACKEND" --compile_warmup "$COMPILE_WARMUP")
