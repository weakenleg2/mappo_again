import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from algorithms.mappo.utils.separated_buffer import SeparatedReplayBuffer
from algorithms.mappo.utils.util import update_linear_schedule
from algorithms.mappo.algorithms.r_mappo.algorithm.ops_utils import compute_clusters
from algorithms.mappo.utils.simple_buffer import ReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

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
        # print(self.episode_length)
        self.n_trajectories = self.all_args.n_trajectories
        self.n_rollout_threads = self.all_args.n_rollout_threads
        # print(self.n_rollout_threads)
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.actor_hidden_size = self.all_args.actor_hidden_size
        self.critic_hidden_size = self.all_args.critic_hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.model_count = min(10,self.all_args.num_agents)

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.pretrain_dur = self.all_args.pretrain_dur
        self.easy_buffer = ReplayBuffer(self.all_args,self.envs.observation_space('agent_0'),
                                       
                                       self.envs.action_space('agent_0')
                                       )
        # env_dict = {
        # "obs": {"shape": self.envs.observation_space('agent_0'), "dtype": np.float32},
        # "rew": {"shape": 1, "dtype": np.float32},
        # "next_obs": {"shape": self.envs.observation_space('agent_0'), "dtype": np.float32},
        # "done": {"shape": 1, "dtype": np.float32},
        # "act": {"shape": self.envs.action_space('agent_0'), "dtype": np.float32},
        # "agent": {"shape": self.num_agents, "dtype": np.float32},
        # }
        # rb = ReplayBuffer(int(self.num_agents * ops['pretraining_steps'] * parallel_envs * n_steps), env_dict)

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
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


        from algorithms.mappo.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.mappo.algorithms.r_mappo.algorithm.rMAPPOPolicy_pre import R_MAPPOPolicy as Policy
            



        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space if self.use_centralized_V else self.envs.observation_space('agent_0')
            # print("self.envs.share_observation_space",self.envs.share_observation_space)
            # policy network
            # we could think here is 32,5
            po = Policy(self.all_args,
                        self.envs.observation_space('agent_0'),
                        share_observation_space,
                        self.envs.action_space('agent_0'),
                        self.model_count,
                        device = self.device)
            po.sample_laac(self.n_rollout_threads)
            if self.all_args.algorithm_mode == 'ops':
                po.laac_sample = torch.zeros(self.n_rollout_threads, self.all_args.num_agents).long()
            self.policy.append(po)


        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space if self.use_centralized_V else self.envs.observation_space('agent_0')
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

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos
    
    # def idx_train(self):
    #     cluster_idx = compute_clusters(rb.get_all_transitions(), 
    #                                    self.all_args.num_agents,self.num_mini_batch, None, 
    #                                    3e-4, 10, 10, 0.0001, self.device)
    #     for agent_id in range(self.num_agents):
    #         self.policy[agent_id].laac_sample= cluster_idx.repeat(self.n_rollout_threads, 1)

    

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