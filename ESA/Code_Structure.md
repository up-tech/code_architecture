## Code structure of ESA

### Overview

![](Images/overview.png)

### Code Structure

<details>
  <summary>Code Structure</summary>

```
├── configs
│   ├── agent.cfg
│   ├── env.config
│   ├── explorer.cfg
│   ├── optimizer.cfg
│   ├── policy.config
│   ├── replay_buffer.cfg
│   └── train.config
├── crowd_env
│   ├── envs
│   │   ├── crowd_env.py
│   │   ├── crowd_sim.py
│   │   ├── env_util.py
│   │   ├── policy
│   │   │   ├── linear.py
│   │   │   ├── orca.py
│   │   │   ├── policy_factory.py
│   │   │   ├── policy.py
│   │   └── utils
│   │       ├── action.py
│   │       ├── agent.py
│   │       ├── human.py
│   │       ├── info.py
│   │       ├── robot.py
│   │       ├── state.py
│   │       └── utils.py
├── crowd_nav
│   ├── first_step.py
│   ├── global_util.py
│   ├── policy
│   │   ├── cadrl.py
│   │   ├── esa.py
│   │   ├── lstm_rl.py
│   │   ├── multi_human_rl.py
│   │   ├── policy_factory.py
│   │   └── sarl.py
│   ├── test.py
│   ├── test_runner.py
│   ├── train.py
│   ├── train_runner.py
│   └── utils
│       ├── explorer.py
│       ├── __init__.py
│       ├── memory2.py
│       ├── plot.py
│       ├── trainer.py
│       └── util.py
├── models
│   ├── esa_model.py
│   ├── lstm_rl_model.py
│   └── sarl_model.py
├── pf_helper
│   ├── agent_builder.py
│   ├── explorer_builder.py
│   ├── network_builder.py
│   ├── optimizer_builder.py
│   ├── pf_runner.py
│   └── replay_buffer_builder.py
├── pfrl
│   ├── action_value.py
│   ├── agent.py
│   ├── agents
│   │   ├── a2c.py
│   │   ├── a3c.py
│   │   ├── acer.py
│   │   ├── al.py
│   │   ├── categorical_double_dqn.py
│   │   ├── categorical_dqn.py
│   │   ├── ddpg.py
│   │   ├── double_dqn.py
│   │   ├── double_pal.py
│   │   ├── dpp.py
│   │   ├── dqn.py
│   │   ├── iqn.py
│   │   ├── pal.py
│   │   ├── ppo.py
│   │   ├── reinforce.py
│   │   ├── soft_actor_critic.py
│   │   ├── state_q_function_actor.py
│   │   ├── td3.py
│   │   └── trpo.py
│   ├── collections
│   │   ├── persistent_collections.py
│   │   ├── prioritized.py
│   │   └── random_access_queue.py
│   ├── distributions
│   │   ├── delta.py
│   ├── env.py
│   ├── envs
│   │   ├── abc.py
│   │   ├── multiprocess_vector_env.py
│   │   └── serial_vector_env.py
│   ├── experiments
│   │   ├── evaluation_hooks.py
│   │   ├── evaluator.py
│   │   ├── hooks.py
│   │   ├── __init__.py
│   │   ├── prepare_output_dir.py
│   │   ├── train_agent_async.py
│   │   ├── train_agent_batch.py
│   │   └── train_agent.py
│   ├── explorer.py
│   ├── explorers
│   │   ├── additive_gaussian.py
│   │   ├── additive_ou.py
│   │   ├── boltzmann.py
│   │   ├── epsilon_greedy.py
│   │   ├── greedy.py
│   ├── functions
│   │   ├── bound_by_tanh.py
│   │   ├── __init__.py
│   │   ├── lower_triangular_matrix.py
│   ├── initializers
│   │   ├── chainer_default.py
│   │   ├── lecun_normal.py
│   ├── nn
│   │   ├── atari_cnn.py
│   │   ├── bound_by_tanh.py
│   │   ├── branched.py
│   │   ├── concat_obs_and_action.py
│   │   ├── empirical_normalization.py
│   │   ├── __init__.py
│   │   ├── lmbda.py
│   │   ├── mlp_bn.py
│   │   ├── mlp.py
│   │   ├── noisy_chain.py
│   │   ├── noisy_linear.py
│   │   ├── recurrent_branched.py
│   │   ├── recurrent.py
│   │   └── recurrent_sequential.py
│   ├── optimizers
│   │   └── rmsprop_eps_inside_sqrt.py
│   ├── policies
│   │   ├── deterministic_policy.py
│   │   ├── gaussian_policy.py
│   │   └── softmax_policy.py
│   ├── policy.py
│   ├── q_function.py
│   ├── q_functions
│   │   ├── dueling_dqn.py
│   │   ├── state_action_q_functions.py
│   │   └── state_q_functions.py
│   ├── replay_buffer.py
│   ├── replay_buffers
│   │   ├── episodic.py
│   │   ├── persistent.py
│   │   ├── prioritized_episodic.py
│   │   ├── prioritized.py
│   │   └── replay_buffer.py
│   ├── testing.py
│   ├── utils
│   │   ├── ask_yes_no.py
│   │   ├── async_.py
│   │   ├── batch_states.py
│   │   ├── clip_l2_grad_norm.py
│   │   ├── conjugate_gradient.py
│   │   ├── contexts.py
│   │   ├── copy_param.py
│   │   ├── env_modifiers.py
│   │   ├── __init__.py
│   │   ├── is_return_code_zero.py
│   │   ├── mode_of_distribution.py
│   │   ├── random.py
│   │   ├── random_seed.py
│   │   ├── recurrent.py
│   │   ├── reward_filter.py
│   │   └── stoppable_thread.py
│   └── wrappers
│       ├── atari_wrappers.py
│       ├── cast_observation.py
│       ├── continuing_time_limit.py
│       ├── __init__.py
│       ├── normalize_action_space.py
│       ├── randomize_action.py
│       ├── render.py
│       ├── scale_reward.py
│       └── vector_frame_stack.py
├── plot_success_rate.py
├── requirements.txt
├── run_dqn.py
├── test_pool.py
└── visualize_test.py
```

</details>

### Environment Setup

#### Observation

self_state:        $s = [d_g, v_{pref}, v_x, v_y, r]$

human_state:   $w_i = [p_x, p_y, v_x, v_y, d^i, r^i, r+r^i]$

**Code**

```python
#file location: crowd_env/envs/utils/robot.py
state = JointState(self.get_full_state(), ob)
```

#### Reward

```math
R_t(s_t^{jn}, a_t) = \begin{cases} -0.25 & \text{if\ $d_t$ < 0} \\ -0.1+d_t/2 & \text{else\ if\ $d_t$ < 0.2} \\ 1 & \text{else\ if\ $p_t$ = $p_g$} \\ 0 & \text{otherwise} \end{cases}
```

<!--

$$R_t(s_t^{jn}, a_t) = \begin{cases} -0.25 & \text{if\ $d_t$ < 0} \\ -0.1+d_t/2 & \text{else\ if\ $d_t$ < 0.2} \\ 1 & \text{else\ if\ $p_t$ = $p_g$} \\ 0 & \text{otherwise} \end{cases}$$

-->

<details>
  <summary>Code</summary>

```python
#file location: Crowd_env/envs/crowd_sim.py
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
    reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
    done = False
    info = Danger(dmin)
else:
    reward = 0
    done = False
    info = Nothing()
```

</details>

#### Action

- The action space consists of 33 discrete actions: 
  1. 4 speeds exponentially spaced between $(0, v_{pref}]$
  2. 8 headings evenly spaced between $(0, 2\pi)$
  3. (0, 0)

#### Terminal condition

- Timeout
- Reaching goal
- Collision

#### Dynamics

**env.step()**

```pseudocode
Input: global arguments, action: a
1 if robot is visible then
2   Get all humans’ state si and robot’ state
3 else
4   Get all humans’ state si
5 end
6 Calculate all humans’ action a i using orca
7 Detection collision between robot and humans
8 Detection collision between humans (just for warning)
9 Check if reaching the goal
10 Calculate reward
11 Check if terminal conditions were satisfied
12 Update robot’s state and humans’s state
13 Get observation ob
Output: ob reward done info
```

### Value Network

![](Images/framework.png)

- Data Process

```python
Spatial_Temporal_Transformer(
  (Embedding): Embedding(
    (embedding): Linear(in_features=13, out_features=128, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  
  (temporal_encoder): Encoder(
    (layers): ModuleList(
      (0): EncoderLayer(
        (attn_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=128, out_features=384, bias=True)
          (out): Linear(in_features=128, out_features=128, bias=True)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (proj_dropout): Dropout(p=0.1, inplace=False)
          (softmax): Softmax(dim=-1)
          (avgpool): AdaptiveAvgPool1d(output_size=1)
        )
        (mlp_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=128, out_features=512, bias=True)
          (fc2): Linear(in_features=512, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (drop_path): DropPath()
      )
    )
    (encoder_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  
  (Spatial_encoder): Encoder(
    (layers): ModuleList(
      (0): EncoderLayer(
        (attn_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=128, out_features=384, bias=True)
          (out): Linear(in_features=128, out_features=128, bias=True)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (proj_dropout): Dropout(p=0.1, inplace=False)
          (softmax): Softmax(dim=-1)
          (avgpool): AdaptiveAvgPool1d(output_size=1)
        )
        (mlp_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=128, out_features=512, bias=True)
          (fc2): Linear(in_features=512, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (drop_path): DropPath()
      )
    )
    (encoder_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  
  (avgpool): AdaptiveAvgPool1d(output_size=1)
  (value_Linear_1): Linear(in_features=134, out_features=256, bias=True)
  (act): GELU(approximate='none')
  (value_Linear_2): Linear(in_features=256, out_features=128, bias=True)
  (value_Linear_3): Linear(in_features=128, out_features=1, bias=True)
  (softmax): Softmax(dim=-1)
)
```

### Training Process

- **main**

```pseudocode
1 Load configuration
2 Implement memory model trainer explorer
3 Set policy as ESA
/* start imitation learning */
4 for episodes = 1, K IL do
5   while not done do
6     Generate action through orca
7     Push state, action, reward in memory
8   end
9 end
10 for epoch = 1, M IL do
11   Get one batch data from memory
12   Calculate loss between reward and value(return through model)
13   Using SGD execute gradient descent and update model’s parameters
14 end
/* start reinforcement learning */
15 Dynamically set epsilon while training
16 for episodes = 1, K RL do
17   while not done do
18     Generate action through sampling and scoring from action space
19     Collect data into memory
20   end
21   Using SGD execute gradient descent
22 end
23 Saving model’s parameters
```

**Details of action generated while RL**

- Random sample action from action space when probability less than epsilon
- Or $a_t = argmax_{a_t\in A}R(s_{t+\Delta t}^{jn},a_t)+{\gamma}^{{\Delta t}\cdot v_{pref}}V(s_{t+\Delta t}^{jn},a_t)$

### Testing Process

```pseudocode
1 Load configuration
2 Set policy as sarl
3 while not done do
4    for action in action_space do
5       Calculate self_state at next time step according to single integrator model
6       Calculate humans_state and next_state_reward at next time step using onestep_lookahead(policy is orca)
7       Concatenate self_state and humans_state as next_state
8       Input next_state into network and get next_state_value
9       Calculate value using next_state_value and next_state_reward
10      Get value and action pair
11    Choose action with highest value from pairs as execute action
11 Update env
12 end
```

- Method to calculate value: $R(s_{t+\Delta t}^{jn},a_t)+{\gamma}^{{\Delta t}\cdot v_{pref}}V(s_{t+\Delta t}^{jn},a_t)$
- Method to pursuance onestep_lookahead: Using action as input to execute step() function in sim env but don't update env state

<!--

**utils for html showing**

```markdown
<details>
  <summary>Code</summary>

</details>
```

-->
