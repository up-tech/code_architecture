### Code structure and simulation

#### structure



**flow chart paste in here** not done yet



- Observation, action and Reward

Those elements defined in drl_vo_nav/drl_vo/src/turtlebot_gym/turtlebot_gym/envs/drl_nav_env.py.

#### Run Training

##### Launch Gazebo environment

- startup script

```bash
$ sh run_drl_vo_policy_training_desktop.sh ~/drl_vo_runs
#create logger files
$ roslaunch drl_vo_nav drl_vo_nav_train.launch log_dir:="${DIR}"
```

- drl_vo_nav_train.launch

```xml
<launch>
  <!-- Pedsim Gazebo -->
  <include file="$(find pedsim_simulator)/launch/robot.launch">
  <!-- AMCL -->
  <include file="$(find robot_gazebo)/launch/amcl_demo_drl.launch">
  <!-- CNN DATA -->
  <include file="$(find drl_vo_nav)/launch/nav_cnn_data.launch"/>
  <!-- DRL-VO Control Policy -->
  <include file="$(find drl_vo_nav)/launch/drl_vo_train.launch">
  <!-- Rviz -->
  <include file="$(find robot_gazebo)/launch/view_navigation.launch" if="$(arg rviz)"/>
</launch>
```

- pedsim_simulator

```xml
<!-- load env model in gazebo -->
<include file="$(find gazebo_ros)/launch/empty_world.launch">
      <arg name="world_name" value="$(arg world_name)"/>
      <arg name="gui" value="$(arg gui)" />
</include>

<!-- this node spawn pedsim actors to gazebo once, then the plugin updates their pose -->  
<!-- load pedestrian model /pkg_path/models/actor_model.sdf -->
<!-- get pedestrain pose from subscribe /pedsim_simulator/simulated_agents topic-->
<node pkg="pedsim_gazebo_plugin" type="spawn_pedsim_agents.py" name="spawn_pedsim_agents"  output="screen">
</node>

<!-- Place gazebo frame at map frame -->
<node pkg="tf" type="static_transform_publisher" name="gazebo_map_broadcaster"
  args="-1 -1 0 0 0 0  odom gazebo 100"/>

<!-- loading urdf, model of turtlebot2 -->
<include file="$(find robot_gazebo)/launch/includes/$(arg base).launch.xml">
</include>

<!-- publish tf according to urdf-->
<node pkg="robot_state_publisher" type="robot_state_publisher" name="robot_state_publisher">
  <param name="publish_frequency" type="double" value="30.0" />
</node>

<!-- main simulator node -->
<!-- run simulation，publish related topics -->
<node name="pedsim_simulator" pkg="pedsim_simulator" type="pedsim_simulator" output="screen">
</node>

<!-- publish marker msg to rviz -->
<include file="$(find pedsim_visualizer)/launch/visualizer.launch"/>
```

- amcl

```xml
<!-- loadlo map -->
<node name="map_server" pkg="map_server" type="map_server" args="$(arg map_file)" />

<!-- using AMCL to achieve localization function of robot in simulation -->
<node pkg="amcl" type="amcl" name="amcl">

<!-- Move base -->
<!-- load global/local costmap -->
<!-- load local planner: dwa  -->
<!-- load global planner: NavfnROS  -->
<include file="$(find robot_gazebo)/launch/includes/move_base_cnn.launch.xml">
<arg name="cmd_vel_topic" value="/move_base/cmd_vel"/>
</include>
```

- CNN data topic(observation message from gazebo to DRL medol)

```xml
<launch>
  <!-- Subgoal Publisher --> <!-- get subgoal from global plan using pursuit -->
  <node name="pure_pursuit" pkg="drl_vo_nav" type="pure_pursuit.py" output="screen" required="true">
  </node>

  <!-- CNN Data  Publisher -->
  <node name="cnn_data_pub" pkg="drl_vo_nav" type="cnn_data_pub.py"/>

  <!-- Robot Pose Publisher -->  <!-- subscribe gloal pose of robot w.r.t. map frame -->
  <node name="robot_pose_pub" pkg="drl_vo_nav" type="robot_pose_pub.py"/>

  <!-- Pedestrian Publisher -->
  <node name="track_ped_pub" pkg="drl_vo_nav" type="track_ped_pub.py" output="screen"/>

  <!-- Goal visualization -->
  <node name="goal_visualize" pkg="drl_vo_nav" type="goal_visualize.py" output="screen" />

</launch>

```

- cnn_data_pub(observation)

```python
#CNN_data msg (self defined topic)
float32[] ped_pos_map		# pedestrian's positon costmap in Cartesian coordinate: 2 channel, 20m * 20m
float32[] scan	  	# 720 range data from the laser scan
float32[] scan_all	# 1080 range data from the laser scan
float32[] image_gray	        # image data from the zed camera
float32[] depth	        # depth image data from the zed camera
float32[] goal_cart     # current goal in robot frame
float32[] goal_final_polar    # final goal in robot frame
float32[] vel                 # current velocity in robot frame

# passing value to CNN_data by callback function
self.ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self.ped_callback)
self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
self.goal_sub = rospy.Subscriber("/cnn_goal", Point, self.goal_callback)

self.vel_sub = rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.vel_callback)

# publish CNN_data topic, including laser_data, pedestrian_kinematics, subgoal_postion
self.cnn_data_pub = rospy.Publisher('/cnn_data', CNN_data, queue_size=1, latch=False)
```

- track_ped_pub.py

```python
#subcribe and publish pedestrain topic (pedestrian pose frame is base_footprint)
self.ped_sub = rospy.Subscriber('/pedsim_visualizer/tracked_persons', TrackedPersons, self.ped_callback)
self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
self.track_ped_pub = rospy.Publisher('/track_ped', TrackedPersons, queue_size=10)
```

- goal_visualize.py

publish marker message of sub_goal to rviz

---

##### Training process

**launch the training process**

- drl_vo_train.launch

```xml
<launch>
  <!-- DRL-VO publisher --> <!-- loading network parameters -->
  <node name="drl_vo_cmd" pkg="drl_vo_nav" type="drl_vo_train.py" output="screen">
    <param name="model_file" value="$(arg model_file)" type="string"/>
    <param name="log_dir" value="$(arg log_dir)" type="string"/>
  </node>

  <!-- Mix cmd vel publisher -->
  <node name="mix_cmd_vel" pkg="drl_vo_nav" type="cmd_vel_pub.py" output="screen" >
    <remap from="cmd_vel" to="teleop_velocity_smoother/raw_cmd_vel"/>
  </node>  

</launch>
```

- drl_vo_train.py

```python
#create environment using Gym

# Create and wrap the environment
env = gym.make('drl-nav-v0')
#drl-nav-v0 is a self defined environment

# policy parameters:
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(pi=[256], vf=[128])]
)

# continue training:
kwargs = {'tensorboard_log':log_dir, 'verbose':2, 'n_epochs':10, 'n_steps':512, 'batch_size':128,'learning_rate':5e-5}
model_file = rospy.get_param('~model_file', "./model/drl_pre_train.zip")
model = PPO.load(model_file, env=env, **kwargs)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
model.learn(total_timesteps=2000000, log_interval=5, tb_log_name='drl_vo_policy', callback=callback, reset_num_timesteps=True)

# Saving final model
model.save("drl_vo_model")
print("Training finished.")
env.close()
```

- gazebo_connection.py

interface between drl-nav-v0 and gazebo, mainly using to pause and unpause sim

- drl_nav_env.py()  

define observation/reward/action

---

#### Experiment

todo: finish this part

---

#### Tips

#### cmd_vel (pub velocity cmd to gazebo)

bold font is node name

**drl_vo_cmd** -> /drl_cmd_vel -> **mix_cmd_vel** -> /teleop_velocity_smoother/raw_cmd_vel -> **mobile_base_nodelet_manager** -> /mobile_base/commands/velocity -> **gazebo**

1. dwa pub cmd_vel as topic name: /move_base/cmd_vel
2. gazebo subscribe topic name is /mobile_base/commands/velocity
3. so if want using cmd genenrate by dwa
4. should modify cmd pub by drl_vo
5. then remap topic publish by move_base to /mobile_base/commands/velocity