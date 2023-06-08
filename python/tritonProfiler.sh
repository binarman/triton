#! /bin/bash

## $1: driver program
## $2: M
## $3: N
## $4: K
## $5: 1: reduced tuning space

if [[ $# -lt 5 ]];then
    echo "Usage: ./tritonProfiler.sh <driver program> M N K <reduceTuningSpace>"
    exit
fi

PROF_RESULT_FILE="results.stats.csv"

DRIVER=$1
M=$2
N=$3
K=$4
reduceSpace=$5

if [[ $reduceSpace -eq 0 ]];then
    ## Tuning space for Triton
#    BLOCK_RANGE=(16 32 64 128)
#    SPLIT_K_RANGE=(2 4 5 6 8 10 12 14 16 18 20 22 24)
#    NUM_WARPS_RANGE=(1 2 4 8)
#    GROUP_M_RANGE=(2 4 6 8 10 12)
    ## Tuning space for rocMLIR
else ## Reduced tuning space
    ## Tuning space for Triton
#    BLOCK_RANGE=(16 32 64 128)
#    SPLIT_K_RANGE=(1 2 5 8 10 12 18 24)
#    NUM_WARPS_RANGE=(1 2 4 8)
#    GROUP_M_RANGE=(4 8 12)
    ## Tuning space for rocMLIR
fi

BLOCK_RANGE=(32 64 128)
SPLIT_K_RANGE=(1)
NUM_WARPS_RANGE=(1)
GROUP_M_RANGE=(1 2)

SMALL_M=0
if [[ $M -le 16 ]];then
    GROUP_M_RANGE=(1)
    SMALL_M=1
fi

echo "Tuning GEMM for M=$M, N=$N, K=$K"

minTime=""
##################################
## Looping BLOCK_M              ##
##################################
for BLOCK_M in ${BLOCK_RANGE[@]}
do
    ## Skip BLOCK_M if it is too large for M
    if [[ $M -le 16 ]] && [[ $BLOCK_M -ne 16 ]]; then
        continue
    fi
    ##################################
    ## Looping GROUP_M              ##
    ##################################
    for GROUP_M in ${GROUP_M_RANGE[@]}
    do
        if [[ ${SMALL_M} -eq 0 ]]; then
            num_block_m=$((M/BLOCK_M))
            ## Skip GROUP_M if it it too large
            if [[ $num_block_m -lt $GROUP_M ]];then
                continue
            fi
        fi
        ##################################
        ## Looping BLOCK_N              ##
        ##################################
        for BLOCK_N in ${BLOCK_RANGE[@]}
        do
            ## Skip BLOCK_N if it is too large for N
            if [[ $N -le 16 ]] && [[ $BLOCK_N -ne 16 ]]; then
                continue
            fi
            ##################################
            ## Looping BLOCK_K              ##
            ##################################
            for BLOCK_K in ${BLOCK_RANGE[@]}
            do
                ##################################
                ## Looping SPLIT_K              ##
                ##################################
                for SPLIT_K in ${SPLIT_K_RANGE[@]}
                do
                    ## Skip SPLIT_K if K % (SPLIT_K * BLOCK_K) != 0
                    leap=$((SPLIT_K * BLOCK_K))
                    mod=$((K%leap))
                    if [[ $mod -ne 0 ]]; then
                        continue
                    fi
                    ##################################
                    ## Looping num_warps            ##
                    ##################################
                    for num_warps in ${NUM_WARPS_RANGE[@]}
                    do
                        perfConfig="$BLOCK_M,$BLOCK_N,$BLOCK_K,$SPLIT_K,$GROUP_M,$num_warps"
                        if [[ ${SMALL_M} -eq 0 ]]; then
                            Msg=$(rocprof --stats python $DRIVER -m $M -n $N -k $K \
                                          -blockM ${BLOCK_M} -blockN ${BLOCK_N} -blockK ${BLOCK_K} \
                                          -num_warps ${num_warps} -splitK ${SPLIT_K} \
                                          -groupM ${GROUP_M})
                        else
                            Msg=$(rocprof --stats python $DRIVER -m $M -n $N -k $K \
                                          -blockM ${BLOCK_M} -blockN ${BLOCK_N} -blockK ${BLOCK_K} \
                                          -num_warps ${num_warps} -splitK ${SPLIT_K})
                        fi

                        time=$(sed -n '/matmul_kernel/p' ${PROF_RESULT_FILE} \
                                   | awk -F ',' '{print $4}')
                        if [[ $minTime == "" ]] || [[ $time -lt $minTime ]];then
                            minTime=$time
                            bestPerfConfig=$perfConfig
                        fi
                        echo "Checked $perfConfig  best parameters: $bestPerfConfig --> $minTime"

                    done
                done
            done
        done
    done
done

echo "Best Result: $M,$N,$K  best parameters: $bestPerfConfig --> $minTime"
