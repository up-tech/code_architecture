### DRL_VO代码分析

---

- Observation, action and Reward

observation等定义在文件drl_vo_nav/drl_vo/src/turtlebot_gym/turtlebot_gym/envs/drl_nav_env.py中，详见下面drl_nav_env.py()函数的说明

---

#### Run Training

##### Gazebo仿真环境相关

- startup script

```bash
$ sh run_drl_vo_policy_training_desktop.sh ~/drl_vo_runs
#创建日志文件夹
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
<!-- gazebo中加载环境模型 -->
<include file="$(find gazebo_ros)/launch/empty_world.launch">
      <arg name="world_name" value="$(arg world_name)"/>
      <arg name="gui" value="$(arg gui)" />
</include>

<!-- this node spawn pedsim actors to gazebo once, then the plugin updates their pose -->  
<!-- 根据 /pkg_path/models/actor_model.sdf 加载人物模型 -->
<!-- 人物的pose 通过订阅 /pedsim_simulator/simulated_agents 话题获得-->
<node pkg="pedsim_gazebo_plugin" type="spawn_pedsim_agents.py" name="spawn_pedsim_agents"  output="screen">
</node>

<!-- Place gazebo frame at map frame -->
<node pkg="tf" type="static_transform_publisher" name="gazebo_map_broadcaster"
  args="-1 -1 0 0 0 0  odom gazebo 100"/>

<!-- 加载turtlebot2的urdf, model, 速度平滑模块 -->
<include file="$(find robot_gazebo)/launch/includes/$(arg base).launch.xml">
</include>

<!-- 根据urdf, 发布tf -->
<node pkg="robot_state_publisher" type="robot_state_publisher" name="robot_state_publisher">
  <param name="publish_frequency" type="double" value="30.0" />
</node>

<!-- main simulator node -->
<!-- 运行仿真，发布相关话题 -->
<node name="pedsim_simulator" pkg="pedsim_simulator" type="pedsim_simulator" output="screen">
</node>

<!-- 发布rviz中的marker相关话题 -->
<include file="$(find pedsim_visualizer)/launch/visualizer.launch"/>
```

- amcl

```xml
<!-- 加载地图 -->
<node name="map_server" pkg="map_server" type="map_server" args="$(arg map_file)" />

<!-- 使用amcl进行仿真中机器人的定位 -->
<node pkg="amcl" type="amcl" name="amcl">

<!-- Move base -->
<!-- 加载global/local costmap -->
<!-- 加载local planner: dwa  -->
<!-- 加载global planner: NavfnROS  -->
<include file="$(find robot_gazebo)/launch/includes/move_base_cnn.launch.xml">
<arg name="cmd_vel_topic" value="/move_base/cmd_vel"/>
</include>
```

- CNN data topic(用于从gazebo中获取obersvation)

```xml
<launch>
  <!-- Subgoal Publisher --> <!-- 通过pursuit截取gloal plan上的subgoal -->
  <node name="pure_pursuit" pkg="drl_vo_nav" type="pure_pursuit.py" output="screen" required="true">
  </node>

  <!-- CNN Data  Publisher -->
  <node name="cnn_data_pub" pkg="drl_vo_nav" type="cnn_data_pub.py"/>

  <!-- Robot Pose Publisher -->  <!-- 订阅map frame下Robot 的 gloal pose -->
  <node name="robot_pose_pub" pkg="drl_vo_nav" type="robot_pose_pub.py"/>

  <!-- Pedestrian Publisher -->
  <node name="track_ped_pub" pkg="drl_vo_nav" type="track_ped_pub.py" output="screen"/>

  <!-- Goal visualization -->
  <node name="goal_visualize" pkg="drl_vo_nav" type="goal_visualize.py" output="screen" />

</launch>

```

- cnn_data_pub(observation)

```python
#CNN_data msg 通过自定义话题发布
float32[] ped_pos_map		# pedestrian's positon costmap in Cartesian coordinate: 2 channel, 20m * 20m
float32[] scan	  	# 720 range data from the laser scan
float32[] scan_all	# 1080 range data from the laser scan
float32[] image_gray	        # image data from the zed camera
float32[] depth	        # depth image data from the zed camera
float32[] goal_cart     # current goal in robot frame
float32[] goal_final_polar    # final goal in robot frame
float32[] vel                 # current velocity in robot frame

# 订阅话题,在回调函数中赋值给CNN_data
self.ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self.ped_callback)
self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
self.goal_sub = rospy.Subscriber("/cnn_goal", Point, self.goal_callback)

self.vel_sub = rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.vel_callback)

# 发布CNN_data话题, 包含laser_data, pedestrian_kinematics, subgoal_postion
self.cnn_data_pub = rospy.Publisher('/cnn_data', CNN_data, queue_size=1, latch=False)
```

- track_ped_pub.py

```python
#订阅并发布pedestrain的相关话题 frame为base_footprint
self.ped_sub = rospy.Subscriber('/pedsim_visualizer/tracked_persons', TrackedPersons, self.ped_callback)
self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
self.track_ped_pub = rospy.Publisher('/track_ped', TrackedPersons, queue_size=10)
```

- goal_visualize.py

发布sub_goal的marker给rviz显示

---

##### training相关

- drl_vo_train.launch

```xml
<launch>
  <!-- DRL-VO publisher --> <!-- 加载网络模型参数 -->
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
#使用gym创建环境

# Create and wrap the environment
env = gym.make('drl-nav-v0')
#这里是定义了对接仿真环境的接口获取数据的方法
#drl-nav-v0为自定义的环境类型

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

drl-nav-v0和gazebo的接口，主要用于pause和unpause sim

- drl_nav_env.py() 定义observation/reward/action

```python
class DRLNavEnv(gym.Env):#继承gym.Env

#observation
def _get_observation(self):
    """
    Returns the observation.
    """
    self.ped_pos = self.cnn_data.ped_pos_map
    self.scan = self.cnn_data.scan
    self.goal = self.cnn_data.goal_cart
    
    self.vel = self.cnn_data.vel

    # ped map:
    # MaxAbsScaler:
    v_min = -2
    v_max = 2
    self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
    self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)

    # scan map:
    # MaxAbsScaler:
    temp = np.array(self.scan, dtype=np.float32)
    scan_avg = np.zeros((20,80))
    for n in range(10):
        scan_tmp = temp[n*720:(n+1)*720]
        for i in range(80):
            scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
            scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])

    scan_avg = scan_avg.reshape(1600)
    scan_avg_map = np.matlib.repmat(scan_avg,1,4)
    self.scan = scan_avg_map.reshape(6400)
    s_min = 0
    s_max = 30
    self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)

    # goal:
    # MaxAbsScaler:
    g_min = -2
    g_max = 2
    self.goal = np.array(self.goal, dtype=np.float32)
    self.goal = 2 * (self.goal - g_min) / (g_max - g_min) + (-1)
    #self.goal = self.goal.tolist()

    # observation:
    self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None) 
    return self.observation

#reward
def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    # reward parameters:
    r_arrival = 20 #15
    r_waypoint = 3.2 #2.5 #1.6 #2 #3 #1.6 #6 #2.5 #2.5
    r_collision = -20 #-15
    r_scan = -0.2 #-0.15 #-0.3
    r_angle = 0.6 #0.5 #1 #0.8 #1 #0.5
    r_rotation = -0.1 #-0.15 #-0.4 #-0.5 #-0.2 # 0.1

    angle_thresh = np.pi/6
    w_thresh = 1 # 0.7

    # reward parts:
    r_g = self._goal_reached_reward(r_arrival, r_waypoint)
    r_c = self._obstacle_collision_punish(self.cnn_data.scan[-720:], r_scan, r_collision)
    r_w = self._angular_velocity_punish(self.curr_vel.angular.z,  r_rotation, w_thresh)
    r_t = self._theta_reward(self.goal, self.mht_peds, self.curr_vel.linear.x, r_angle, angle_thresh)
    reward = r_g + r_c + r_t + r_w #+ r_v # + r_p
    #rospy.logwarn("Current Velocity: \ncurr_vel = {}".format(self.curr_vel.linear.x))
    rospy.logwarn("Compute reward done. \nreward = {}".format(reward))
    return reward

#step
def step(self, action):
    """
    Gives env an action to enter the next state,
    obs, reward, done, info = env.step(action)
    """
    # Convert the action num to movement action
    self.gazebo.unpauseSim()
    self._take_action(action)
    self.gazebo.pauseSim()
    obs = self._get_observation()
    reward = self._compute_reward()
    done = self._is_done(reward)
    # self._done_pub.publish(done)
    info = self._post_information()
    #print('info=', info, 'reward=', reward, 'done=', done)
    return obs, reward, done, info
```

- custom_cnn_full.py(定义网络结构)

```python
#网络初始化
#observation_space 通过 spaces.Box限定在（-1，1）
def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 256):
```

