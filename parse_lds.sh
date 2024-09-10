#!/usr/bin/bash

for i in $(find ~/.triton/cache/  -name streamk_gemm_*.json ); do
  echo -n "$i " | sed 's/.*\/streamk_gemm/streamk_gemm/' | sed 's/\.json//'
  sed  's/.*shared": \([0-9]*\).*/\1/' < $i
  echo
done > shared_sizes

for i in $(find ~/.triton/cache/  -name streamk_gemm_*.json ); do
  SIZE=$(sed  's/.*shared": \([0-9]*\).*/\1/' < $i)
  if (( $SIZE > 65536 )); then
    echo -n  "$i " | sed 's/.*\/streamk_gemm/streamk_gemm/' | sed 's/\.json//'
    echo $SIZE
  fi
done > failed_configs
