import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from algorithms.mappo.utils.separate_buffer_dpo import SeparatedReplayBuffer
from algorithms.mappo.utils.util import update_linear_schedule
from gymnasium import spaces
from algorithms.mappo.utils.simple_buffer import ReplayBuffer
import math

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, most_common_index,config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_trajectories = self.all_args.n_trajectories
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.actor_hidden_size = self.all_args.actor_hidden_size
        self.critic_hidden_size = self.all_args.critic_hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        # self.com_p = 0.05

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.pretrain_dur = self.all_args.pretrain_dur
        self.vae_lr = self.all_args.vae_lr
        self.kl = self.all_args.vae_kl
        self.vae_epoch = self.all_args.vae_epoch
        self.clusters = math.ceil(self.all_args.num_agents/2)
        self.vae_zfeatures = self.all_args.vae_zfeatures
        self.vae_batch = self.all_args.vae_batchsize
        self.mid_gap = self.all_args.mid_gap
        self.easy_buffer = ReplayBuffer(self.all_args,self.envs.observation_space('agent_0'),
                                       
                                       self.envs.action_space('agent_0')
                                       )
        
        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            self.save_dir = str(self.run_dir / 'models')
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)


        from algorithms.mappo.algorithms.r_mappo.r_mappo_dpo_simp import R_MAPPO as TrainAlgo
        from algorithms.mappo.algorithms.r_mappo.algorithm.rMAPPOPolicy_dpo_sep import R_MAPPOPolicy as Policy
        # print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space('agent_0'))
        print("action_space: ", self.envs.action_space)

        self.policy = []
        agent_classifications = most_common_index

        # Dictionary to store policies for each class
        class_policies = {}
        
        for agent_id in range(self.num_agents):
            # print()
            agent_class = agent_classifications[agent_id]
            if agent_class not in class_policies:
                share_observation_space = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(
                    self.num_agents*self.envs.observation_space('agent_0').shape[0],
                ),  # 24 is the observation space of each walker, 3 is the package observation space
                dtype=np.float32, ) if self.use_centralized_V else self.envs.observation_space('agent_0')
            
            # print("self.envs.share_observation_space",self.envs.share_observation_space('agent_0'))
            # policy network
                class_policy = Policy(self.all_args,
                            self.envs.observation_space('agent_0'),
                            share_observation_space,
                            self.envs.action_space('agent_0'),
                            device = self.device)
                class_policies[agent_class] = class_policy

            self.policy.append(class_policies[agent_class])

        print(self.policy)
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # print()
            share_observation_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(
                self.num_agents*self.envs.observation_space('agent_0').shape[0],
            ),  # 24 is the observation space of each walker, 3 is the package observation space
            dtype=np.float32,
            ) if self.use_centralized_V else self.envs.observation_space('agent_0')
            # buffer
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space('agent_0'),
                                       share_observation_space,
                                       self.envs.action_space('agent_0'))
            self.buffer.append(bu)
            self.trainer.append(tr)

        if self.model_dir is not None:
            self.restore()
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
            next_penalty,_ = self.trainer[agent_id].policy.get_penalty(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_penalty[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_penalty = _t2n(next_penalty)
            self.buffer[agent_id].compute_penalty(next_penalty, self.trainer[agent_id].value_normalizer)
    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            torch.save(self.trainer[0].policy.init_dict, str(self.save_dir) + "/init.pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        infos = {k:{} for k in train_infos[0].keys()}
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                infos[k][str(agent_id)] = v
        # for k in infos.keys():
        #     self.writter.add_scalars(k, infos[k], total_num_steps)
        for k in infos.keys():
            if self.use_wandb:
                wandb.log({k: infos[k]}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, infos[k], total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)