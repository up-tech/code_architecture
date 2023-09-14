## Network

---

### RNN

#### Principle

<img src="Images/rnn.png" style="zoom:50%;" />

**Forward**

$s_t = \phi(Ux_t + Ws_{t-1})$

$o_t = f(Vs_t)$

$s_t$: the state value of hidden layer at time t

$o_t$: output value of time t

$U$: weights of input x

$W$: weights of input state $s_{t-1}$ at time t

$\phi$: activation function

$V$: weights of output layer

$f$: activation function

**loss function**

cross entropy: $\sum_{i=1}^{T}{-\overline{o}_t\log{o_t}}$

**Backward**

BPTT(back-propagation through time)

---

### LSTM

#### Principle

**RNN**

<img src="Images/rnn_module.png" style="zoom:50%;" />

**LSTM**

<img src="Images/lstm_module.png" style="zoom:50%;" />

<img src="Images/symbols.png" style="zoom:33%;" />

- Pointwise Operation: dot product

**Cell State**

<img src="Images/cell_state.png" style="zoom:33%;" />

**Gate**

<img src="Images/gate.png" style="zoom:25%;" />

- Contains a sigmod layer and a dot product

- It determines how much message can go through

**Forget Gate**

<img src="Images/forget_gate.png" style="zoom:33%;" />

- Reduce value towards 0

$f_t=\sigma(W_f\cdot[h_{t-1}, x_t]+b_f)$

**Input Gate**

<img src="Images/input_gate.png" style="zoom:33%;" />

- Determine whether to ignore input or not

$i_t=\sigma(W_i\cdot[h_{t-1},x_t]+b_i$

$$\widetilde{C}_t = \tanh{(W_C\cdot[h_{t-1},x_t]+b_C)}$$

**Cell State**

<img src="Images/update_cell_state.png" style="zoom:33%;" />

$C_t=f_t\cdot{C_{t-1}}+i_t\cdot{\widetilde{C}_t}$

**Output Gate**

<img src="Images/output_gate.png" style="zoom:33%;" />

- Determine whether to use hidden state

$o_t = \sigma(W_o\cdot[h_{t-1},x_t])+b_o$

$h_t = o_t\cdot\tanh{(C_t)}$

#### Implementation using Pytorch

