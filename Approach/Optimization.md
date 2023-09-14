## Optimization

### Reference about optimization

#### Optimization target

- Value based: optimize all action values of a state as accurate as possible（价值函数为基础）
- Policy Gradient: optimize actions to states as accurate as possible（动作差分为基础）

#### Pattern mapping

- on policy
- off policy

### A3C(Asynchronous Advantage Actor Critic)

#### Key features

- A2C(Advantage Actor Critic): single thread or synchronous
- A3C: multi thread
- Critic output value
- Actor output mean and standard devidation
- Asynchronous: open multi thread workers to update parameters randomly
- Advantage: Bootstrap values from last state
- Actor: Decide action vector (vector)
- Critic: Evaluate action value (a real number)

#### Every learn step

- Push gradients to global network
- Pull weights from global network

### PPO(Proximal Policy Optimization)

#### Key features

- PPO from TRPO
- Training method: Actor-critic/stochastic model
- Define loss with ratio of adjacent policies
- Restrict gradients with clip or KL penalty
- PP0/A3C can share A/C networks

#### How to asynchronize work

- Open multiple workers to expand searching space with environments
- Exchange learning experience by push to/pull from global network

#### Every learn step

1. Generate actions from local network
2. Pull weights from global network
3. Push gradients to global network

#### PPO loss function

$$L_t(\theta) = \min[\frac{\pi(a \mid s)}{\pi_{old}(a \mid s)}A_{\pi_{old}}(s,a), clip(\frac{\pi(a \mid s)}{\pi_{old}(a \mid s)}A_{\pi_{old}}(s,a),1-\epsilon, 1+\epsilon)]$$

- there are two policy probability functions $\pi(a)$ and $\pi_{old}(a)$
- let $\pi(a)$ decided by global actor
- let $\pi_{old}(a)$ by old (local) actor networks





