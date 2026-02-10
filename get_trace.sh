#!/usr/bin/bash

rm -rf sm_trace
ROCPROF_ATT_LIBRARY_PATH=/rocprof-decoder/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib/ rocprofv3 --att -i att.json -d sm_trace -- python skinny_matmul/utils/03-matrix-multiplication.py
tar -cf sm_trace.tar sm_trace/ui_output*
mv sm_trace.tar /host/
