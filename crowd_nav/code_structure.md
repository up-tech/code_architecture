## Code structure of CrowdNav

### Overview



### Environment Setup

#### Env

```mermaid
flowchart TD
    id1(collect all humans observation)-->id2(generate humans action);
    id2(generate humans action)-->id3(check collision);
    id3(check collision)-->id4(check goal reached);
    id4(check goal reached)-->id5(calculate reward & check terminal);
    id5(calculate reward & check terminal)-->id6(update robot and humans state);
    id6(update robot and humans state)-->id7(get next observation);
```

##### Step

```python
#file location: CrowdNav/crowd_sim/envs/utils/crowd_sim.py

def step(self, action, update=True):
    """
    Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

    """
    human_actions = []
    for human in self.humans:
        # observation for humans is always coordinates
        ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
        if self.robot.visible:
            ob += [self.robot.get_observable_state()]
        human_actions.append(human.act(ob))

    # collision detection
    dmin = float('inf')
    collision = False
    for i, human in enumerate(self.humans):
        px = human.px - self.robot.px
        py = human.py - self.robot.py
        if self.robot.kinematics == 'holonomic':
            vx = human.vx - action.vx
            vy = human.vy - action.vy
        else:
            vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
            vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
        ex = px + vx * self.time_step
        ey = py + vy * self.time_step
        # closest distance between boundaries of two agents
        closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
        if closest_dist < 0:
            collision = True
            # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
            break
        elif closest_dist < dmin:
            dmin = closest_dist

    # collision detection between humans
    human_num = len(self.humans)
    for i in range(human_num):
        for j in range(i + 1, human_num):
            dx = self.humans[i].px - self.humans[j].px
            dy = self.humans[i].py - self.humans[j].py
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
            if dist < 0:
                # detect collision but don't take humans' collision into account
                logging.debug('Collision happens between humans in step()')

    # check if reaching the goal
    end_position = np.array(self.robot.compute_position(action, self.time_step))
    reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

    if self.global_time >= self.time_limit - 1:
        reward = 0
        done = True
        info = Timeout()
    elif collision:
        reward = self.collision_penalty
        done = True
        info = Collision()
    elif reaching_goal:
        reward = self.success_reward
        done = True
        info = ReachGoal()
    elif dmin < self.discomfort_dist:
        # only penalize agent for getting too close if it's visible
        # adjust the reward based on FPS
        reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
        done = False
        info = Danger(dmin)
    else:
        reward = 0
        done = False
        info = Nothing()

    if update:
        # store state, action value and attention weights
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values.append(self.robot.policy.action_values)
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights.append(self.robot.policy.get_attention_weights())

        # update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step
        for i, human in enumerate(self.humans):
            # only record the first time the human reaches the goal
            if self.human_times[i] == 0 and human.reached_destination():
                self.human_times[i] = self.global_time

        # compute the observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
    else:
        if self.robot.sensor == 'coordinates':
            ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

    return ob, reward, done, info
```

##### Reset

```python
#file location: CrowdNav/crowd_sim/envs/utils/crowd_sim.py

def reset(self, phase='test', test_case=None):
    """
    Set px, py, gx, gy, vx, vy, theta for robot and humans
    :return:
    """
    if self.robot is None:
        raise AttributeError('robot has to be set!')
    assert phase in ['train', 'val', 'test']
    if test_case is not None:
        self.case_counter[phase] = test_case
    self.global_time = 0
    if phase == 'test':
        self.human_times = [0] * self.human_num
    else:
        self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
    if not self.robot.policy.multiagent_training:
        self.train_val_sim = 'circle_crossing'

    if self.config.get('humans', 'policy') == 'trajnet':
        raise NotImplementedError
    else:
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
            if phase in ['train', 'val']:
                human_num = self.human_num if self.robot.policy.multiagent_training else 1
                self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
            else:
                self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError

    for agent in [self.robot] + self.humans:
        agent.time_step = self.time_step
        agent.policy.time_step = self.time_step

    self.states = list()
    if hasattr(self.robot.policy, 'action_values'):
        self.action_values = list()
    if hasattr(self.robot.policy, 'get_attention_weights'):
        self.attention_weights = list()

    # get current observation
    if self.robot.sensor == 'coordinates':
        ob = [human.get_observable_state() for human in self.humans]
    elif self.robot.sensor == 'RGB':
        raise NotImplementedError

    return ob
```

#### Observation

$p_x,p_y,v_x,v_y,radius$

p<sub>x</sub> : human's position x
p<sub>y</sub> : human's position y
v<sub>x</sub> : human's velocity x
v<sub>y</sub> : human's velocity y
radius : human's radius

- Source of observation data: Ground truth in gym environment
- Collect each human's $p_x,p_y,v_x,v_y,radius$ and form the observation

```python
#file location: CrowdNav/crowd_sim/envs/utils/agents.py

def get_next_observable_state(self, action):
    self.check_validity(action)
    pos = self.compute_position(action, self.time_step)
    next_px, next_py = pos
    if self.kinematics == 'holonomic':
        next_vx = action.vx
        next_vy = action.vy
    else:
        next_theta = self.theta + action.r
        next_vx = action.v * np.cos(next_theta)
        next_vy = action.v * np.sin(next_theta)
return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)
```

#### Reward



```python
#file location: CrowdNav/crowd_sim/envs/utils/crowd_sim.py

if self.global_time >= self.time_limit - 1:
    reward = 0
    done = True
    info = Timeout()
elif collision:
    reward = self.collision_penalty
    done = True
    info = Collision()
elif reaching_goal:
    reward = self.success_reward
    done = True
    info = ReachGoal()
elif dmin < self.discomfort_dist:
    # only penalize agent for getting too close if it's visible
    # adjust the reward based on FPS
    reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
    done = False
    info = Danger(dmin)
else:
    reward = 0
    done = False
    info = Nothing()
```

#### Action

#### Terminal

### Network Structure

### Training Process

```mermaid
---
title: Training
---

flowchart TD
  configure["path
  logging
  environment"]
  
  load_config["env.config
  policy.config
  train.config"]
  
  il["imitation_learning"]
  
  rl["reinforcement_learning"]
  
  trainer["configure trainer"]
  
  explorer["configure explorer"]
  
  configure --> load_config;
  load_config --> trainer;
  trainer --> explorer;
  explorer --> il;
  il --> rl;
  
```



### Testing Process