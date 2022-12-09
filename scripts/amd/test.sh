# clear

set -x

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# pytest -rfs --verbose python/test | tee $LOG_DIR/test_all.log
pytest -rfs --verbose python/test/unit/language/test_core.py 2>&1 | tee $LOG_DIR/test_core.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_bin_op 2>&1 | tee $LOG_DIR/test_bin_op.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_shift_op 2>&1 | tee $LOG_DIR/test_shift_op.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_tensor_atomic_rmw 2>&1 | tee $LOG_DIR/test_tensor_atomic_rmw.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_permute 2>&1 | tee $LOG_DIR/test_permute.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_dot 2>&1 | tee $LOG_DIR/test_dot.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_atomic_rmw 2>&1 | tee $LOG_DIR/test_atomic_rmw.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_atomic_cas 2>&1 | tee $LOG_DIR/test_atomic_cas.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_f8_xf16_roundtrip 2>&1 | tee $LOG_DIR/test_f8_xf16_roundtrip.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_reduce1d 2>&1 | tee $LOG_DIR/test_reduce1d.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_reduce2d 2>&1 | tee $LOG_DIR/test_reduce2d.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_masked_load_shared_memory 2>&1 | tee $LOG_DIR/test_masked_load_shared_memory.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_libdevice_tensor 2>&1 | tee $LOG_DIR/test_libdevice_tensor.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_libdevice_scalar 2>&1 | tee $LOG_DIR/test_libdevice_scalar.log
# pytest -rfs --verbose python/test/unit/language/test_dequantize.py 2>&1  | tee $LOG_DIR/test_dequantize.log
# pytest -rfs --verbose python/test/unit/language/test_random.py 2>&1  | tee $LOG_DIR/test_random.log
# pytest -rfs --verbose python/test/unit/operators/test_blocksparse.py 2>&1  | tee $LOG_DIR/test_blocksparse.log
# pytest -rfs --verbose python/test/unit/operators/test_cross_entropy.py 2>&1  | tee $LOG_DIR/test_cross_entropy.log
# pytest -rfs --verbose python/test/unit/operators/test_matmul.py 2>&1  | tee $LOG_DIR/test_matmul.log
# pytest -rfs --verbose python/test/unit/runtime/test_cache.py 2>&1  | tee $LOG_DIR/test_cache.log
# pytest -rfs python/test/regression/test_performance.py 2>&1 | tee $LOG_DIR/test_performance.log