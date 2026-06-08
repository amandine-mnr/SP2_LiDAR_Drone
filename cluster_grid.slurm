#!/bin/bash -l

# To run grid search on the cluster

# ----------------------------
# Configurations for you job
# ----------------------------
#SBATCH --job-name=drone_loc_job #give your job a name for the squeue
#SBATCH --partition=super-fat #keep this as is
#SBATCH --time=2-00:00:00 #time after which the job automatically ends d-hh:mm:ss
#SBATCH --nodes=1 #how many nodes you will be needing
#SBATCH --nodelist=disalcs[9]#which node(s) you want to use, alternatively you can say
#which node(s) you want to exclude with --exclude=disalcs[2]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G #how much memory
#SBATCH --gres=gpu:2 #how many gpus

##SBATCH --output=/scratch/%u/%x_%j.out
##SBATCH --error=/scratch/%u/%x_%j.err
#SBATCH --output=outputs/slurm_%j.out #where to save logs
#SBATCH --error=outputs/slurm_%j.err #where to save errors
# ----------------------------
# Beginning of your demands
# ----------------------------

set -euo pipefail
mkdir -p /scratch/$USER
echo "=== [$(date)] Job $SLURM_JOB_ID started on $(hostname) ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES:-?}"

# ----------------------------------------------------------------------------
# Run everything inside Docker
# ----------------------------------------------------------------------------
sudo -n docker run --rm -i \
  --runtime=nvidia \
  --network=host \
  --ipc=host \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /scratch:/scratch:rw \
  -v "$HOME/micromamba:$HOME/micromamba:rw" \
  -e HOME="$HOME" \
  -e USER="$USER" \
  -e NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}" \
  -u "$(id -u):$(id -g)" \
  -v "$HOME:$HOME:rw" -w "$HOME" \
  -v /usr/local/bin/micromamba:/usr/local/bin/micromamba:ro \
  --entrypoint bash disal/gz:jazzy-harmonic -lc '

    set -e

    echo "=== Inside container on $(hostname) ==="

    PROJECT_DIR="$HOME/ros2_ws"
    cd "$PROJECT_DIR"

    # Clean old folders
    rm -rf build install log

    # Source ROS
    source /opt/ros/jazzy/setup.bash

    # Python venv
    export PATH="/usr/local/bin:$PATH"
    export MAMBA_ROOT_PREFIX="$HOME/micromamba"
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate SP_venv
    echo "Virtual environment activated"
    which python
    python --version


    # Build inside container
    colcon build \
      --cmake-args -DPython3_EXECUTABLE=$(which python)
    echo "Colcon build"

    # Source the workspace
    source install/setup.bash
    echo "ROS2 workspace sourced"

    echo "[INFO] Working directory: $PROJECT_DIR"

    # CPU / threading control
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    # Set Gazebo resource path-
    export GZ_SIM_RESOURCE_PATH="$HOME/crazyflie-simulation/simulator_files/gazebo/"
    echo "GZ_SIM_RESOURCE_PATH set to $GZ_SIM_RESOURCE_PATH"

    # Start ROS2 - Gazebo bridges
    ros2 run ros_gz_bridge parameter_bridge \
      --ros-args -p config_file:=$HOME/ros2_ws/bridge.yaml &
    BRIDGE_PID=$!
    sleep 5

    mkdir -p "$PROJECT_DIR/grid_results"
    echo "Starting grid search"
    python "$HOME/grid_search.py"
    echo "Grid search finished"

    sleep 10

    #-------------------

    echo "=== Container finished ==="
'

echo "=== [$(date)] Job finished ==="