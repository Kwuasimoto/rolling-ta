Splicing LSTM Embeddings into PPO Rollouts

This section describes how to cleanly integrate (splice) LSTM forecast embeddings into PPO observations during rollout collection and optimization.

You can paste this into a separate markdown file.

1. Goal

Augment PPO’s observation with future-aware information from a pre-trained (or jointly trained) LSTM forecast model.

At each timestep t, PPO receives:

obs_t = concat(obs_base_t, z_t)


Where:

obs_base_t = indicators + agent state

z_t = LSTM latent embedding at time t

2. Online Rollout Splicing (Inference Time)

At each environment step:

Build base observation:

obs_base_t: (N, F_base)


Feed same base features into LSTM:

x_t: (1, N, F_base)
→ z_t: (1, N, Z)


Squeeze time dim:

z_t_squeezed: (N, Z)


Concatenate:

obs_t = cat([obs_base_t, z_t_squeezed], dim = -1)


Result:

obs_t: (N, F_base + Z)


Store in rollout buffer at index t.

3. Rollout Buffer Storage

Over T steps, store:

obs_base: (T, N, F_base)
z_roll:   (T, N, Z)


Optionally store only the concatenated form:

obs: (T, N, F)


But keeping them separate is useful for debugging and ablations.

4. PPO Update Time

During PPO optimization:

If stored separately:

obs = cat([obs_base, z_roll], dim = -1)
→ (T, N, F)


Then slice sequences per minibatch:

obs_mb: (T, B_mb, F)


Feed directly into the PPO policy LSTM.

No recomputation of LSTM forecasts is needed during PPO updates.

5. Hidden State Handling

You maintain two independent recurrent states:

Forecast LSTM:

Hidden state per env

Updated every step during rollout

Reset on episode boundary

Policy LSTM (PPO):

Hidden state per env

Stored at rollout start

Replayed during PPO updates

They are never mixed.

6. Detachment Rules

LSTM embeddings z_t are treated as constants for PPO:

No gradient flows back into forecast LSTM

Detach before concatenation:

z_t = z_t.detach()


Unless you explicitly want joint training.

7. Synchronization Requirement

It is critical that:

The features used to compute obs_base_t

Are identical to those fed into the LSTM at time t

This ensures:

Temporal alignment

No information leakage

Stable policy learning

8. Asset Batching

If running multiple assets:

N = num_envs = num_assets × envs_per_asset


Splicing is done per batch row:

obs_t[i] = concat(obs_base_t[i], z_t[i])


Hidden states are maintained per row.

9. Common Pitfalls

Avoid:

Mixing time indices (off-by-one errors)

Recomputing LSTM outputs during PPO update

Shuffling sequences before LSTM replay

Letting PPO gradients flow into forecast LSTM

Using different normalizations between LSTM and PPO

10. Result

With correct splicing:

PPO receives future-aware latent context

Rollouts remain on-policy

Recurrent PPO training remains stable

Forecast model can evolve independently