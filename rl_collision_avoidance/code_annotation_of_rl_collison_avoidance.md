### rl_collision_avoidance代码流程分析

#### 代码结构

---

- Stage环境
```
├── stage_ros-add_pose_and_crash
│   ├── rviz
│   │   └── stage.rviz
│   ├── src
│        └── stageros.cpp
├── suquare_test.py
└── worlds
    ├── circle.world
    ├── rink.png
    ├── stage1.world
    ├── stage2.world
```

- 主要代码
```
├── circle_test.py
├── circle_world.py
├── model
│   ├── __init__.py
│   ├── net.py
│   ├── ppo.py
│   └── utils.py
├── policy
│   ├── stage1_1.pth
│   ├── stage1_2.pth
│   └── stage2.pth
├── ppo_stage1.py
├── ppo_stage2.py
├── single_agent_test.py
├── single_agent_world.py
├── stage_world1.py
├── stage_world2.py
```

#### 训练过程（stage1）

##### Observation

```python
# 代码路径：rl-collision-avoidance/ppo_stage1.py

# 关键函数：get_laser_observation()
#         get_local_goal()
#         get_self_speed()
obs = env.get_laser_observation()
obs_stack = deque([obs, obs, obs])
goal = np.asarray(env.get_local_goal())
speed = np.asarray(env.get_self_speed())
state = [obs_stack, goal, speed] #observation

#代码路径：rl-collision-avoidance/stage_world1.py
def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 6.0 - 0.5 #normalize scan data

def get_self_speed(self):
    return self.speed

def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]
```

##### Reward and Terminal

```python
# 代码路径：rl-collision-avoidance/ppo_stage1.py

#关键函数:get_reward_and_terminate()
#input: step_num
#output: reward/terminal/terminal type
r, terminal, result = env.get_reward_and_terminate(step)

#代码路径：rl-collision-avoidance/stage_world1.py
def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0
        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        if np.abs(w) >  1.05:
            reward_w = -0.1 * np.abs(w)

        if t > 150:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result
```

##### Train

- Network architecture Setup

```python
# 代码路径：rl-collision-avoidance/model/net.py
class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):
        super(CNNPolicy, self).__init__()
        #nn.Parameter():创建可训练的参数，这些参数会在模型巡训练过程中自动更新
        #参数初始化为[0, 0]
        self.logstd = nn.Parameter(torch.zeros(action_space))
        
        #actor network
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        #critic network
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)
```

- Run

```python
# 代码路径：rl-collision-avoidance/ppo_stage1.py

# 关键函数：StageWorld()
#         CNNPolicy()

#初始化agent环境
#input: 一帧激光的点数、MPI线程的编号(rank)、总agents的个数
#output: 第rank个环境env
#env的作用：获得obervation和reward
env = StageWorld(512, index=rank, num_env=NUM_ENV)

#初始化网络结构
#input: LASER_HIST = 3 action_space = 2
#output: policy
policy = CNNPolicy(frames=LASER_HIST, action_space=2)

#Adam(adaptive moment estimation)适应性矩估计
#一阶矩控制模型更新方向
#二阶矩控制步长（学习率）
opt = Adam(policy.parameters(), lr=LEARNING_RATE)

#关键函数：run()
run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)

#函数主要内容：
def run(comm, env, policy, policy_path, action_bound, optimizer):
    #仿真最大运行次数
    for id in range(MAX_EPISODES):
        env.reset_pose()
        #获取observation
        state = [obs_stack, goal, speed]
        while not terminal and not rospy.is_shutdown():
            state_list = comm.gather(state, root=0)
            # input: env\observation\policy
            # output: value\action\action_prob\scaled_action
            # 传递给仿真环境执行的为scaled_action
            # 函数详细说明在下面PPO部分
            v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # execute actions
            real_action = comm.scatter(scaled_action, root=0)
            env.control_vel(real_action)
            # get informtion
            # 详细见上述reward and terminal部分
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1

            # get next state
            state_next = [obs_stack, goal_next, speed_next]
            #收集HORIZON(default:128)次数据后更新state
            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)

            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    # batch size为128
                    # 关键函数transform_buffer()
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    # 关键函数generate_train_data()
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    #关键函数ppo_update_stage1()
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1

            step += 1
            state = state_next

#关键函数ppo_update_stage1()
# 还没看懂
def ppo_update_stage1(policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=128, num_env=8, frames=3, obs_size=24, act_size=4):
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    obss = obss.reshape((num_step*num_env, frames, obs_size))
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))

    print('update')
```

- PPO

```python
# 代码路径：rl-collision-avoidance/ppo.py

# 关键函数：generate_action()
def generate_action(env, state_list, policy, action_bound):
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0]) #scan
            goal_list.append(i[1]) #local_goal
            speed_list.append(i[2]) #self_speed

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)
        
        #numpy数组转化为tensor放入gpu进行计算
        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
        #调用net.py中定义的forward()函数
        #输入observation
        #输出value, action ,action_prob, action_mean
        v, a, logprob, mean = policy(s_list, goal_list, speed_list)
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        #约束action在action_bound内，线速度(0.0, 1.0) 角速度(-1.0, 1.0)
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

# 关键函数：generate_action()
# 还没看懂
def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs
```

- Forward

```python
# 代码路径：rl-collision-avoidance/net.py

# 关键函数：forward()
# input: observation\local goal\self speed
def forward(self, x, goal, speed):
    """
        returns value estimation, action, log_action_prob
    """
    # action
    a = F.relu(self.act_fea_cv1(x))
    a = F.relu(self.act_fea_cv2(a))
    a = a.view(a.shape[0], -1)
    a = F.relu(self.act_fc1(a))

    a = torch.cat((a, goal, speed), dim=-1)
    a = F.relu(self.act_fc2(a))
    #两个(128,1)的全连接层
    #激活函数分别为sigmod funchtion和hyperbolic tangent function
    #约束输出在(0.0, 1.0)和(-1.0, 1,0)
    #完成input observation maps to vector
    mean1 = F.sigmoid(self.actor1(a))
    mean2 = F.tanh(self.actor2(a))
    mean = torch.cat((mean1, mean2), dim=-1)
    #将输出赋值给可训练的参数logstd
    logstd = self.logstd.expand_as(mean)
    std = torch.exp(logstd)
    #final action为平均值为mean，标准差为std的正态分布的采样
    action = torch.normal(mean, std)

    # action prob on log scale
    # 调用utils.py计算采样的高斯密度
    logprob = log_normal_density(action, mean, std=std, log_std=logstd)

    # value
    # 价值网络的结构和action网络一致，但只有一维输出
    v = F.relu(self.crt_fea_cv1(x))
    v = F.relu(self.crt_fea_cv2(v))
    v = v.view(v.shape[0], -1)
    v = F.relu(self.crt_fc1(v))
    v = torch.cat((v, goal, speed), dim=-1)
    v = F.relu(self.crt_fc2(v))
    v = self.critic(v)

    return v, action, logprob, mean
```

---

#### 测试过程
