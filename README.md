# SP2_LiDAR_Drone
EPFL, DISAL, SP2, 2026  
Project goals : 
- Design a localization algorithm to localize a Crazyflie drone using a fixed LiDAR  
- Implement a simulation framework (Gazebo Sim, ROS2) to test it  
- Compare the performance of the algorithm with varying LiDAR and drone trajectory parameters (grid search)  

## Versions
ROS2 version : Jazzy  
Gazebo Sim version : Harmonic

## To launch a single run :

source /opt/ros/jazzy/setup.bash  
colcon build  
source install/setup.bash  
ros2 launch start launch.py  

--> A csv file is created in the folder metrics_log/ with all the logs

For multiple drones (3 drones): Same "ros2 launch start launch.py", but on the branch "multiple_drones" of the git

## To launch a grid search :

source /opt/ros/jazzy/setup.bash  
colcon build  
source install/setup.bash  
python grid_search.py  

--> A csv file is created for each run of the grid in the folder metrics_log/, as well as a recap of the different runs' performance written in grid_search_results.csv


## Explanations of the different files :

grid_search.py :  
Script to perform grid search  

cluster_docker.slurm :  
Slurm file to launch one simulation on the cluster  

cluster_grid.slurm :  
Slurm file to launch the grid search on the cluster  

rviz_config.rviz :  
Configuration file, used by the supervisor node to automatically open Rviz2  

### ros2_ws/ :

bridge.yaml :   
Configuration file for the bridges between ROS2 and Gazebo Sim  

param.yaml :  
Configuration file used when launching one simulation  

crazyflie package :  
(controller_node) controller.py : to control the drone with the keyboard (not used)  
(traj_controller_node) traj_controller.py : to control the drone to follow automatically an oval trajectory  

lidar package :  
kalman.py : functions used for the Kalman filter  
(localization_node) localization.py : main localization algorithm  
metrics.py : script to compute and plot metrics for a single run  

start package :  
(supervisor_node) supervisor.py : handle experiment duration, send stop signal to other nodes  
launch.py : ROS2 launch file for a single experiment  
launch_grid.py : ROS2 launch file for grid search (automatically ran by grid_search.py)  
launch_grid_cluster.py : ROS2 launch file for grid search on the cluster (automatically ran by grid_search.py)  

### Simulation files
crazyflie-simulation/ :  
Contains world sdf files, LiDAR model, drone model, tree models (for the background)  
Drone model from https://github.com/bitcraze/crazyflie-simulation.git  
Tree models from https://github.com/osrf/gazebo_models.git  
lidar_3D_template.sdf : template used by the grid_search script to rewrite the LiDAR sdf file  
lidar_3D.sdf : LiDAR model used when launching a single run  
crazyflie_world_main.sdf : Main world, used for single drone experiments  
crazyflie_world.sdf : World with some hardcoded paths for the cluster (used in launch_grid_cluster.py)  
crazyflie_world_multiple_drones.sdf : World with 3 drones, used in the branch multiple_drones of the git  

### CSV logs :  
results.zip/ :  
grid_search_results_g[i].csv : Recap of each run's metrics and parameters for grid search [i]  
all_grid_g.csv : Concatenation of all 3 grid_search_results_g[i].csv  
metrics_log_g1/ : Grid search results with drone start pos = (0, 0)  
metrics_log_g2/ : Grid search results with drone start pos = (0, -4)  
metrics_log_g3/ : Grid search results with drone start pos = (0, -8)  




