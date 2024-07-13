# vsgp_pcl
Light-Weight Pointcloud Representation with Variational Sparse Gaussian Process

## Building
- Clone the repository as your workspace
```
git clone https://github.com/mahmoud-a-ali/vsgp_pcl.git
```
- Compile it using `catkin build`
```
cd vsgp_pcl
catkin build
source devel/setup.bash
```

## Usage
1. Launch the simulated `cpr_inspection` environment which contains lake, pipe, grass, and tunnel
```
roslaunch jackal_gazebo cpr_inspection.launch 
```
2. Run occupancy surface which convert pointcloud to an occupancy surface 
```
roslaunch vsgp_gen_mdl occ_surface.launch 
```
3. Check that only the `base` node uses GPU where the `scout` node uses CPU
```
# check that the following line is commented for `base` node and uncommented for `scout` node
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU
```
4. Run the `scout` node which trains an VSGP and publish only the inducing points to the `base` node
```
roslaunch vsgp_gen_mdl scout.launch
```
5. Run the `base` node which construct back the pointcloud using the inducing points send by the `scout` node
```
roslaunch vsgp_gen_mdl base.launch
```
#### If using two PCs, one as the Scout robot and the other as the Base station
1. Configure the two PCs to communicate to each other using ROS. you can make either Scout or Base as the `ROS_MASTER`
- On the Scout robot, update `.bashrc` file by adding
```
export ROS_MASTER_URI=http://Scout_IP:11311
export ROS_HOSTNAME=Scout_IP
```
- On the Base station, update `.bashrc` file by adding
```
export ROS_MASTER_URI=http://Scout_IP:11311
export ROS_HOSTNAME=Base_IP
```
- Note: this configuration making the scout as the `ROS_MASTER`
2. On the Scout PC run the following:
```
roslaunch jackal_gazebo cpr_inspection.launch
roslaunch vsgp_gen_mdl occ_surface.launch
roslaunch vsgp_gen_mdl scout.launch
```
3. On the Base PC run the following:
```
roslaunch vsgp_gen_mdl base.launch
```

#### If using real robot as the Scout 
- Replace the command to launch the simulator with the launch file that runs the robot hardware including LiDAR driver and localization node
- Update `pcl_topic` and `localization_topic` in the `occ-surface.launch` file




## Paper: [Light-Weight Pointcloud Representation with Variational Sparse Gaussian Process](https://arxiv.org/pdf/2301.11251)
## Citation
```
@inproceedings{ali2023light,
  title={Light-weight pointcloud representation with sparse gaussian process},
  author={Ali, Mahmoud and Liu, Lantao},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4931--4937},
  year={2023},
  organization={IEEE}
}
```
## video
[![IMAGE](video.png)](https://www.youtube.com/watch?v=CA8HWRIo5KY)
