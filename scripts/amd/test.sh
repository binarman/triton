# clear

set -x

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

pytest -rfs --verbose python/tests | tee $LOG_DIR/test_all.log
# pytest -rfs --verbose "python/tests/test_core.py::test_math_op" | tee $LOG_DIR/test_all.log
# pytest -rfs --verbose "python/tests/test_vecadd.py"

