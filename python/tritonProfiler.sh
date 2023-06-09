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

BLOCK_RANGE=(32 64)
SPLIT_K_RANGE=(1 8)
NUM_WARPS_RANGE=(1 2)

SMALL_M=0
if [[ $M -le 32 ]];then
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
    if [[ $M -le 32 ]] && [[ $BLOCK_M -ne 32 ]]; then
        continue
    fi
    ##################################
    ## Looping BLOCK_N              ##
    ##################################
    for BLOCK_N in ${BLOCK_RANGE[@]}
    do
        ## Skip BLOCK_N if it is too large for N
        if [[ $N -le 32 ]] && [[ $BLOCK_N -ne 32 ]]; then
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
                    echo "rocprof --stats python $DRIVER -m $M -n $N -k $K -blockM ${BLOCK_M} -blockN ${BLOCK_N} -blockK ${BLOCK_K} -num_warps ${num_warps} -splitK ${SPLIT_K}"
                    Msg=$(rocprof --stats python $DRIVER -m $M -n $N -k $K \
                                  -blockM ${BLOCK_M} -blockN ${BLOCK_N} -blockK ${BLOCK_K} \
                                  -num_warps ${num_warps} -splitK ${SPLIT_K})

                    time=$(sed -n '/matmul_kernel/p' ${PROF_RESULT_FILE} \
                               | awk -F ',' '{print $4}')
		    # rm stat file to prevent it spoiling next runs in case they crash
		    rm ${PROF_RESULT_FILE}
                    if [[ $minTime == "" ]] || [[ $time -lt $minTime ]];then
                        minTime=$time
                        bestPerfConfig=$perfConfig
                    fi
                    echo "Checked $perfConfig time: $time best parameters: $bestPerfConfig --> $minTime"

                done
            done
        done
    done
    
done

echo "Best Result: $M,$N,$K  best parameters: $bestPerfConfig --> $minTime"
