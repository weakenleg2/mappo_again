import os
import imageio
import argparse
from custom_envs.mpe import simple_spread_c_v2
import numpy as np
import torch
from algorithms.mappo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

def dict_to_tensor(d):
  d = list(d.values())
  d = np.array(d)
  d = torch.tensor(d)
  return d

def preprocess_obs(obs):
  obs = dict_to_tensor(obs)
  return obs

def get_logits(obs, policies, rnn_state):
  logits = []
  new_rnn_state = []
  for i, p in enumerate(policies):
    o = obs[i].unsqueeze(0)
    l, _, r = p(o, rnn_state[i], torch.tensor(0))
    logits.append(l.squeeze(0))
    new_rnn_state.append(r.squeeze(0))
  
  logits = torch.stack(logits)
  new_rnn_state = torch.stack(new_rnn_state)
  return logits, new_rnn_state

def get_actions(obs, env, rnn_state, policies, training=False):
  actions = {}
  if len(policies) > 1:
    logits, rnn_state = get_logits(obs, policies, rnn_state)
  else:
    logits, _, rnn_state = policies[0](obs, rnn_state, torch.tensor(0))

  logits = np.clip(logits, -1, 1)
  for i, agent in enumerate(env.possible_agents):
    actions[agent] = logits[i]

  return actions, logits, rnn_state

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('filename', description='Path to folder containing actor.pt files', ntype=str)
  parser.add_argument('-n', '--n_agents', type=int, default=3)
  parser.add_argument('-r', '--random_actions', action='store_true')
  parser.add_argument('-f', '--full_com', action='store_true')
  args = parser.parse_args()

  if not hasattr(args, 'hidden_size'):
    args.hidden_size = 256
    args.layer_N = 2
    args.gain = 0.01
    args.use_orthogonal = False
    args.use_policy_active_masks = True
    args.use_naive_recurrent_policy = False
    args.use_recurrent_policy = False
    args.recurrent_N = 0
    args.use_feature_normalization = True
    args.use_ReLU = False 
    args.stacked_frames= 1
    args.use_stacked_frames = False
    args.use_centralized_V = True
    args.share_policy = True
    args.pop_art = False 

  env = simple_spread_c_v2.parallel_env( 
                                      N=args.n_agents,
                                      local_ratio = 0.5, 
                                      max_cycles=25, 
                                      full_comm = args.full_com,
                                      continuous_actions=True,
                                      render_mode = 'rgb_array')

  if not args.random_actions:
    init_dict = torch.load(args.filename + '/init.pt')
    print(init_dict)
    actor_files = [f for f in os.listdir(args.filename) if f.startswith('actor_agent') and f.endswith('.pt')]

    if len(actor_files) == 0 and os.path.isfile(os.path.join(args.filename, 'actor.pt')):
        actor_files = ['actor.pt']
    policies = []
    for actor_file in actor_files :
      state_dict = torch.load(os.path.join(args.filename, actor_file))
    
      policy = R_Actor(init_dict, obs_space=env.observation_space('agent_0'), action_space=env.action_space('agent_0'))
      policy.load_state_dict(state_dict)
      policies.append(policy)

    rnn_state = torch.zeros(args.n_agents, init_dict.recurrent_N, init_dict.actor_hidden_size * 2) 

  tot_reward = 0
  seeds = range(3)

  frames = []
  for s in seeds:
    s = s + 15
    np.random.seed(s)
    torch.manual_seed(s)

    seed_reward = np.zeros(args.n_agents)
    obs = env.reset()
    obs = preprocess_obs(obs)

    while env.agents:
      if args.random_actions:
        actions = {}
        for n in range(args.n_agents):
          a = env.action_space('agent_0').sample()
          a = (*a[0], a[1])
          actions['agent_{}'.format(n)] = a
      else:
        actions, logits, rnn_state = get_actions(obs, env, rnn_state, policies, False)

      next_obs, rewards, dones, truncations, infos = env.step(actions)
      next_obs = preprocess_obs(next_obs)
      obs = next_obs
      rewards = dict_to_tensor(rewards)
      seed_reward = seed_reward + rewards.squeeze().numpy()

      image = env.render()
      frames.append(image)

    tot_reward += seed_reward.mean()

  print(tot_reward / len(seeds))
  writer = imageio.get_writer('gifs' + '/render.mp4', fps=24)

  for frame in frames:
    writer.append_data(frame)
  writer.close()
  env.close()
