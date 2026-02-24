#!/bin/bash
set -x
python3 skinny_matmul/kernels/v0_torch.py
python3 skinny_matmul/kernels/v1_dot2d_mma.py
python3 skinny_matmul/kernels/v2_dot2d_fma.py
python3 skinny_matmul/kernels/v3_dot3d.py
python3 skinny_matmul/kernels/v4_gluon_dot3d.py
python3 skinny_matmul/kernels/v5_gluon_dot3d_local_b.py
