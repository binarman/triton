# clear
pytest -rfs --verbose python/test | tee /dockerx/triton/test_all.log
# pytest -rfs --verbose python/test/unit/language/test_core.py | tee /dockerx/triton/test_core.log
# pytest -rfs --verbose python/test/unit/language/test_dequantize.py | tee /dockerx/triton/test_dequantize.log
# pytest -rfs --verbose python/test/unit/language/test_random.py | tee /dockerx/triton/test_random.log
# pytest -rfs --verbose python/test/unit/operators/test_blocksparse.py | tee /dockerx/triton/test_blocksparse.log