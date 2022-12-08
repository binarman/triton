# clear
# pytest -rfs --verbose python/test/unit/language/test_core.py | tee /dockerx/triton/test_core.log
pytest --capture=tee-sys -rfs --verbose python/test/unit/language/test_core.py::test_masked_load | tee /dockerx/triton/test_masked_load.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_masked_load_shared_memory | tee /dockerx/triton/test_masked_load_shared_memory.log
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_load_cache_modifier
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_reduce1d | tee /dockerx/triton/test_reduce1d.log
