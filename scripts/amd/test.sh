# clear

set -x

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# pytest -rfs --verbose python/test | tee $LOG_DIR/test_all.log
pytest -rfs python/test/regression/test_performance.py 2>&1 | tee $LOG_DIR/test_performance.log
pytest -rfs --verbose python/test/unit/language/test_core.py 2>&1 | tee $LOG_DIR/test_core.log
pytest -rfs --verbose python/test/unit/language/test_dequantize.py 2>&1  | tee $LOG_DIR/test_dequantize.log
pytest -rfs --verbose python/test/unit/language/test_random.py 2>&1  | tee $LOG_DIR/test_random.log
pytest -rfs --verbose python/test/unit/operators/test_blocksparse.py 2>&1  | tee $LOG_DIR/test_blocksparse.log
pytest -rfs --verbose python/test/unit/operators/test_cross_entropy.py 2>&1  | tee $LOG_DIR/test_cross_entropy.log
pytest -rfs --verbose python/test/unit/operators/test_matmul.py 2>&1  | tee $LOG_DIR/test_matmul.log
pytest -rfs --verbose python/test/unit/runtime/test_cache.py 2>&1  | tee $LOG_DIR/test_cache.log
