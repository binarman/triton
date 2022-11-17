set -x

cd python
pip uninstall -y triton

export TRITON_USE_ROCM=ON
# export TRITON_ROCM_DEBUG=ON
export MI_GPU_ARCH=gfx90a

pip install --verbose -e .
pip install -U matplotlib pandas filelock
