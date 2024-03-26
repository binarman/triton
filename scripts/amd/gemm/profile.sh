#!/usr/bin/bash

/sqtt-rocprof-install/bin/rocprof  -i job.txt -d OUT_464  $(which python3) ./one_config.py  --config_str M16_N8192_K128_BM16_BN64_BK64_GM1_SK1_nW4_nS0_EU0_kP1_mfma464
/sqtt-rocprof-install/bin/rocprof  -i job.txt -d OUT_16  $(which python3) ./one_config.py  --config_str M16_N8192_K128_BM16_BN64_BK64_GM1_SK1_nW4_nS0_EU0_kP1_mfma16
