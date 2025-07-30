#!/usr/bin/bash
rm -r output
rocprofv3 --kernel-trace -d output --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_LDS_UNALIGNED_STALL -- python3 microbench_op0.py | tee configs
grep kernel_config_ $(find output -name *counter_collection.csv) |cut -f 9,13,15,16,17 -d, |grep -v SQ_LDS_ADDR_CONFLICT|sed "s/kernel_config_//" > bench_results.csv
echo "results are in bench_results.csv"
