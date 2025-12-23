# Efficient Batching of Recurrent PPO Rollouts

Rust + tch Implementation Guide

This document describes how to collect, batch, and train on rollouts for a recurrent PPO Actor-Critic model using GRU/LSTM in Rust with the tch crate. It is designed for large-scale time series environments (e.g., ~12.6M minute candles across BTC/ETH/LTC).

---

## 1. Why Recurrent PPO Requires Special Batching

In feedforward PPO, samples are i.i.d.

In recurrent PPO, policy outputs depend on:
- Current state `s_t`
- Hidden state `h_t` from previous timesteps

Therefore:
- We must preserve temporal order within sequences
- We must reset hidden states at episode boundaries
- We must batch sequences, not individual transitions

---

## 2. Key Concepts

| Term | Meaning |
|------|---------|
| `N_env` | Number of parallel environments |
| `T_rollout` | Steps collected per environment before update |
| `T_seq` | Sequence length for BPTT (≤ T_rollout) |
| `B` | Number of sequences per minibatch |
| `H` | Hidden size of GRU/LSTM |

**Typical values:**

```
N_env     = 8-32
T_rollout = 128-256
T_seq     = 32-128
H         = 128
```

---

## 3. Rollout Tensor Shapes

During collection:

```
states      : [T_rollout, N_env, S]
actions     : [T_rollout, N_env]
log_probs   : [T_rollout, N_env]
rewards     : [T_rollout, N_env]
dones       : [T_rollout, N_env]
values      : [T_rollout, N_env]
h_states    : [T_rollout, N_env, H]   // stored BEFORE step
```

After flattening:

```
Total steps = T_rollout * N_env
```

But for recurrent training, we do not fully flatten. We keep time.

---

## 4. Rollout Collection Loop

At each step:
1. Pass `(state_t, h_t)` to policy
2. Get `(action_t, logp_t, value_t, h_{t+1})`
3. Step environment
4. Store everything

**Pseudo-code:**

```rust
for t in 0..T_rollout {
    let (action, logp, value, h_next) =
        policy.forward(state, h);

    let (next_state, reward, done) = env.step(action);

    buffer.store(t, state, action, logp, value, reward, done, h);

    h = if done { policy.zero_hidden() } else { h_next };
    state = next_state;
}
```

**Important:**
- Store `h_t`, not `h_{t+1}`
- Reset hidden state on `done`

---

## 5. Advantage and Return Computation

After rollout, compute GAE:

```
advantages[t] = δ_t + γλ(1 - done_t) * advantages[t+1]
returns[t]    = advantages[t] + values[t]
```

Shapes remain: `[T_rollout, N_env]`

---

## 6. Sequence Construction for Recurrent Batching

We now slice rollouts into contiguous sequences of length `T_seq`.

**Number of sequences per env:**

```
K = T_rollout / T_seq
```

**Total sequences:**

```
N_seq = N_env * K
```

**Each sequence has shape:**

```
states_seq   : [T_seq, S]
actions_seq  : [T_seq]
logp_seq     : [T_seq]
adv_seq      : [T_seq]
ret_seq      : [T_seq]
value_seq    : [T_seq]
done_seq     : [T_seq]
h0_seq       : [H]   // hidden at start of sequence
```

### 6.1 Building Sequences

For env `e` and chunk `k`:

```
t_start = k * T_seq
t_end   = (k + 1) * T_seq
```

Take:
- `states[t_start:t_end, e]`
- `h_states[t_start, e]` → `h0_seq`

---

## 7. Shuffling and Minibatching

We shuffle **sequences**, not timesteps.

```
indices = randperm(N_seq)
```

Then group into minibatches of size `B`:

```
num_minibatches = N_seq / B
```

Each minibatch contains:

```
states_mb  : [B, T_seq, S]
actions_mb : [B, T_seq]
logp_mb    : [B, T_seq]
adv_mb     : [B, T_seq]
ret_mb     : [B, T_seq]
h0_mb      : [1, B, H]  // for GRU/LSTM
```

---

## 8. Training Forward Pass

For each minibatch:
1. Initialize hidden state with `h0_mb`
2. Forward through recurrent policy over full sequence
3. Get:
   - `logp_new[t]`
   - `entropy[t]`
   - `value_new[t]`

**In tch:**

```rust
let (pi_logits, values, _) =
    policy.forward_seq(states_mb, h0_mb);
```

**Shapes:**

```
pi_logits : [B, T_seq, A]
values    : [B, T_seq]
```

---

## 9. PPO Loss Computation (Masked)

Flatten time and batch:

```
[B * T_seq, ...]
```

Optionally mask padded steps (if any):

```
mask = 1 - done_mb
```

Then compute:

### Policy Loss

```
ratio = exp(logp_new - logp_old)
L_clip = -mean(min(
    ratio * adv,
    clip(ratio, 1-ε, 1+ε) * adv
))
```

### Value Loss

```
L_v = mean((values_new - returns)^2)
```

### Entropy Bonus

```
L_ent = mean(entropy)
```

### Total Loss

```
L = L_clip + c_v * L_v - c_ent * L_ent
```

Backprop and step optimizer.

---

## 10. Hidden State Handling Rules

| Do | Don't |
|----|-------|
| Store `h_t` at each timestep | Never carry hidden state across minibatches |
| Use `h0_seq` when training each sequence | Never shuffle individual timesteps |
| Reset hidden state when `done == true` or new episode/asset stream starts | Use `h_{t+1}` instead of `h_t` as sequence start |

---

## 11. Typical Hyperparameters

| Parameter | Value |
|-----------|-------|
| `N_env` | 16 |
| `T_rollout` | 256 |
| `T_seq` | 64 |
| `B` | 32 |
| PPO epochs | 3-5 |
| γ | 0.99 |
| λ | 0.95 |
| clip ε | 0.2 |
| value coef | 0.5 |
| entropy coef | 0.001-0.01 |

This yields:

```
N_seq = (256 / 64) * 16 = 64 sequences
→ 2 minibatches of size 32
```

---

## 12. Memory Layout Recommendation

Store rollout buffers as:

```
Vec<Tensor> or single Tensor shaped:
[T_rollout, N_env, ...]
```

Avoid per-step allocations. Preallocate once.

---

## 13. Rust Buffer Struct Sketch

```rust
pub struct RolloutBuffer {
    pub states: Tensor,    // [T, N, S]
    pub actions: Tensor,   // [T, N]
    pub logp: Tensor,      // [T, N]
    pub rewards: Tensor,   // [T, N]
    pub dones: Tensor,     // [T, N]
    pub values: Tensor,    // [T, N]
    pub h_states: Tensor,  // [T, N, H]
}
```

Then slice into sequences via tensor views.

---

## 14. Common Pitfalls

| Pitfall | Why It's Bad |
|---------|--------------|
| Shuffling timesteps instead of sequences | Breaks temporal dependencies |
| Not resetting hidden state on `done` | Leaks information across episodes |
| Using `h_{t+1}` instead of `h_t` as sequence start | Off-by-one hidden state |
| Too long `T_seq` | Unstable gradients |
| Too short `T_seq` | Recurrent core becomes useless |
| Not normalizing advantages | Training instability |

---

## 15. Performance Tips for Large Data

For ~12.6M steps:

- Use parallel envs per asset
- Keep tensors on GPU end-to-end
- Minimize host↔device copies
- Use GRU instead of LSTM for speed
- Log only aggregates, not per-step

---

## 16. Summary

Efficient recurrent PPO batching means:

1. Collect `[T_rollout, N_env]` rollouts with hidden states
2. Slice into contiguous sequences of length `T_seq`
3. Shuffle **sequences**, not timesteps
4. Train with stored initial hidden states
5. Backprop through time within each sequence only

This preserves temporal dependencies while maintaining PPO's minibatch efficiency.

> This batching scheme is critical to making recurrent PPO stable and scalable for multi-million-step crypto time series learning.