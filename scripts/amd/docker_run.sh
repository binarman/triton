set -o xtrace

alias drun='sudo docker run -it --rm --network=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

DEVICES="--gpus 0"
# DEVICES="--device=/dev/kfd --device=/dev/dri"

MEMORY="--ipc=host --shm-size 16G"

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR='/root/$(basename $(pwd))'
WORK_DIR="/dockerx/$(basename $(pwd))"

IMAGE_NAME=nvcr.io/nvidia/pytorch:22.11-py3
# IMAGE_NAME=rocm/pytorch:latest # latest doesnot work
# IMAGE_NAME=rocm/pytorch:rocm4.3.1_ubuntu18.04_py3.6_pytorch_1.10.0
# IMAGE_NAME=triton_rocm_20-52 # build this docker before running

CONTAINER_NAME=triton

# start new container
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
CONTAINER_ID=$(drun -d -w $WORK_DIR --name $CONTAINER_NAME $MEMORY $VOLUMES $DEVICES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
# docker cp . $CONTAINER_ID:$WORK_DIR
# docker exec $CONTAINER_ID bash -c "bash scripts/amd/run.sh"
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
