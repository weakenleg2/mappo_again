import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='rmappo', choices=["rmappo", "mappo"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=2,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='MPE', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--full_comm", action='store_true', help="if agents have full communication")
    parser.add_argument("--com_ratio", type=float, default=0.05, help="Ratio for agent communication penalty,0.05 default for multiwalker,0.1 for simple spread")
    parser.add_argument("--local_ratio", type=float, default=0.5, help="Ratio for agent rewards")
    parser.add_argument("--delay", type=int, default=0, help="delay frames.")
    parser.add_argument("--packet_drop_prob", type=int, default=0, help="drop prob")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=25, help="Max length for any episode")
    parser.add_argument("--n_trajectories", type=int,
                        default=5, help="Number of trajectories to sample per thread")
    # make a try here

    # network parameters
    parser.add_argument("--share_policy", action='store_true',
                        default=False, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=False, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--actor_hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actornetworks") 
    parser.add_argument("--critic_hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for critic networks") 
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Dimension of hidden layers for mlp networks")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    # 此处做了简单改变
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--sigmoid_gain", type=float, default=1.,
                        help="The gain # of sigmoid")
    parser.add_argument("--cnn_layers_params", type=str, default=None,
                        help="The parameters of cnn layer")
    parser.add_argument("--use_maxpool2d", action='store_true',
                        default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--activation_id", type=int,
                        default=1, help="choose 0 to use tanh, 1 to use relu, 2 to use leaky relu, 3 to use elu")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--entropy_lr", type=float, default=5e-4,
                        help='entropy learning rate (default: 5e-4)')
    parser.add_argument("--kl_lr", type=float, default=5e-2,
                        help='kl learning rate (default: 5e-4)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_policy_vhead",
                        action='store_true', default=False, help="by default, do not use policy vhead. if set, use policy vhead.")
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--policy_value_loss_coef", type=float,
                        default=1, help='policy value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default= 0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--inner_max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.98,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    parser.add_argument("--inner_clip_param", type=float, default=0.,
                        help='inner ppo clip parameter (default: 0.2)')
    parser.add_argument("--dual_clip_coeff", type=float, default=3,
                        help='ppo dual clip parameter (default: 3.)')
    parser.add_argument("--IGM_coef", type=float, default=0.01,
                        help='IGM term coefficient (default: 0.01)')
    parser.add_argument("--kl_coef", type=float, default=0.001,
                        help='KL term coefficient (default: 0.001)')
    parser.add_argument("--num_v_out", default=1, type=int, help="number of value heads in critic")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")
    parser.add_argument("--render", action='store_true', help="if Environment should be rendered during training")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    parser.add_argument("--algorithm_mode", type=str, default='ops', help="by default None. set the path to pretrained model.")
    parser.add_argument("--pretrain_dur", type=int, default=40, help="pretrainsteps.")
    parser.add_argument("--vae_lr", type=float, default=1e-5, help="lr for vae.")
    parser.add_argument("--clusters", type=int, default=3, help="clusters.")
    parser.add_argument("--vae_epoch", type=int, default=10, help="epoch for vae")
    parser.add_argument("--vae_zfeatures", type=int, default=10, help="features for vae")
    parser.add_argument("--vae_kl", type=float, default=0.0001, help="kl for vae")
    parser.add_argument("--vae_batchsize", type=int, default=512, help="batchsize for vae")
    parser.add_argument("--mid_gap", type=int, default=560, help="mid gap for vae")

     # attn parameters
    parser.add_argument("--use_attn", action='store_true', default=False, help=" by default False, use attention tactics.")
    parser.add_argument("--attn_N", type=int, default=1, help="the number of attn layers, by default 1")
    parser.add_argument("--attn_size", type=int, default=64, help="by default, the hidden size of attn layer")
    parser.add_argument("--attn_heads", type=int, default=2, help="by default, the # of multiply heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="by default 0, the dropout ratio of attn layer.")
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True, help="by default True, use average pooling for attn model.")
    parser.add_argument("--use_attn_internal", action='store_false', default=True, help="by default True, whether to strengthen own characteristics")
    parser.add_argument("--use_cat_self", action='store_false', default=True, help="by default True, whether to strengthen own characteristics")
    parser.add_argument("--attention_lr", type=float, default=5e-4,
                        help='attention learning rate (default: 5e-4)')
    
    # temporal parameters
    parser.add_argument("--use_graph", action='store_true', default=False, help=" by default False, use temporal graph.")
    parser.add_argument("--mix_id", type=int,
                        default=0, help="choose 0 to use mixer, 1 to use hyper, 2 to use attention")
    parser.add_argument("--train_sim_seq", type=int,
                        default=0, help="choose 0 to train seq agent first, 1 to train seq epoch first, 2 to train sim")
    parser.add_argument("--token_factor", type=float,
                        default=0.5, help="default is 1")
    parser.add_argument("--channel_factor", type=float,
                        default=4, help="default is 1")
    parser.add_argument('--max_edges', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--time_channels', type=int, default=100)
    parser.add_argument('--time_gap', type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help='gumble softmax temperature')
    parser.add_argument("--use_action_attention",  action='store_true', default=True,
                        help='action attention')
    parser.add_argument("--use_vtrace",  action='store_false', default=True,
                        help='use_vtrace')
    parser.add_argument("--skip_connect",  action='store_false', default=True,
                        help='skip connection (default: True)')
    parser.add_argument("--automatic_kl_tuning",  action='store_true', default=False,
                        help='Automaically adjust kl_coef (default: False)')
    parser.add_argument("--automatic_entropy_tuning",  action='store_true', default=False,
                        help='Automaically adjust entropy_coef (default: False)')
    parser.add_argument("--automatic_target_entropy_tuning",  action='store_true', default=False,
                        help='Automaically adjust target entropy (default: False)')
    parser.add_argument("--exponential_avg_discount", type=float, default=0.9,
                        help='exponential average discount')
    parser.add_argument("--exponential_var_discount", type=float, default=0.99,
                        help='exponential variance discount')
    parser.add_argument("--target_entropy_discount", type=float, default=0.9,
                        help='target entropy discount')
    parser.add_argument("--standard_deviation_threshold", type=float, default=0.05,
                        help='standard deviation threshold')
    parser.add_argument("--average_threshold", type=float, default=0.01,
                        help='average threshold')
    parser.add_argument("--threshold", type=float, default=1.,
                        help='tradoff between bpta and mappo')
    parser.add_argument("--decay_factor", type=float, default=1.5,
                        help='tradoff between bpta and mappo')
    parser.add_argument("--agent_layer", type=int, default=1,
                        help='stacked agent layer')
    parser.add_argument("--random_train", action='store_true', default=False,
                        help='')
    parser.add_argument("--decay_id", type=int,
                        default=0, help="choose 0 to use linear_decay, 1 to use cos_decay, 2 to use step_decay")
    # parser.add_argument("--linear_decay", type=args_str2bool, default=False, 
    #                     help='')
    # parser.add_argument("--cos_decay", type=args_str2bool, default=True, 
    #                     help='')
    # parser.add_argument("--step_decay", type=args_str2bool, default=False, 
    #                     help='')
    parser.add_argument("--bc", action='store_false', default=True,
                        help='')
    parser.add_argument("--bc_epoch", type=int, default=15,
                        help='')
    parser.add_argument("--mix_std_x_coef", type=float, default=1.0,
                        help='')
    parser.add_argument("--mix_std_y_coef", type=float, default=0.5,
                        help='')
    # ppg parameters
    parser.add_argument("--aux_epoch", type=int, default=5,
                        help='number of auxiliary epochs (default: 4)')
    parser.add_argument("--clone_coef", type=float, default=1.0,
                        help='clone term coefficient (default: 0.01)')
    parser.add_argument("--use_single_network", action='store_true',
                        default=False, help="Whether to use centralized V function")
    #dpo parameters
    parser.add_argument('--NP_dp_eps', type=float, default=0.5)
    parser.add_argument('--NP_recalc_optimal_V', type=bool, default=False)
    parser.add_argument('--NP_gamma_correct_omega', type=bool, default=False)
    parser.add_argument('--check_V_details', type=bool, default=False)
    parser.add_argument('--check_optimal_V_bound', type=bool, default=False)
    parser.add_argument('--NP_delta', type=float, default=1e-3)
    parser.add_argument('--NP_delta_mode', type=str, default='fix')
    parser.add_argument('--NP_add_delta', type=bool, default=False)
    parser.add_argument('--episode_longer', type=bool, default=False)
    parser.add_argument('--reward_scale_const', type=float, default=None)

    parser.add_argument('--NP_grad_check', type=bool, default=False)
    parser.add_argument('--NP_auto_lr', type=float, default=1e-3)
    parser.add_argument('--NP_auto_target', type=float, default=None)
    parser.add_argument('--NP_clip_coeff_refine', type=bool, default=False)
    parser.add_argument('--NP_use_clip', type=bool, default=False)
    parser.add_argument('--NP_balance_rate', type=float, default=1.0)
    parser.add_argument('--NP_decay_mode', type=str, default='linear')
    parser.add_argument('--NP_decay_rate', type=float, default=0.5)
    parser.add_argument('--NP_coeff_decay', type=bool, default=False)
    parser.add_argument('--NP_coeff_end', type=float, default=None)
    parser.add_argument('--NP_dist_name', type=str, default='H')
    parser.add_argument('--NP_coeff', type=float, default=1e-2)
    parser.add_argument('--new_penalty_method', type=bool, default=False)
    parser.add_argument('--dpo_check_kl_baseline', type=bool, default=False)
    parser.add_argument('--dpo_policy_div_agent_num', type=bool, default=False)
    parser.add_argument('--idv_beta', type=bool, default=False)
    parser.add_argument('--unit_sight_range', type=int, default=9)
    parser.add_argument('--es_judge', type=float, default=0.25)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--correct_kl', type=bool, default=False)
    parser.add_argument('--sp_update_policy', type=str, default='hard')
    parser.add_argument('--inner_refine', action='store_false', default=True)
    parser.add_argument('--para_lower_bound', type=float, default=0.001)
    parser.add_argument('--para_upper_bound', type=float, default=1000.0)
    parser.add_argument('--penalty_beta_sqrt_type', type=str, default='adaptive')
    parser.add_argument('--penalty_beta_type', type=str, default='adaptive')
    parser.add_argument('--no_kl', type=bool, default=False)
    parser.add_argument('--no_sqrt_kl', type=bool, default=False)
    parser.add_argument('--check_kl_output', type=bool, default=False)
    parser.add_argument('--penalty_method', type=bool, default=False)
    parser.add_argument('--beta_kl', type=float, default=1.0)
    parser.add_argument('--dtar_kl', type=float, default=0.02)
    parser.add_argument('--kl_para1', type=float, default=1.5)
    parser.add_argument('--kl_para2', type=float, default=2.0)

    parser.add_argument('--beta_sqrt_kl', type=float, default=1.0)
    parser.add_argument('--dtar_sqrt_kl', type=float, default=0.01)
    parser.add_argument('--sqrt_kl_para1', type=float, default=1.5)
    parser.add_argument('--sqrt_kl_para2', type=float, default=2.0)

    parser.add_argument('--sp_use_q', type=bool, default=False)
    parser.add_argument('--sp_check', type=bool, default=False)
    parser.add_argument('--sp_clip', type=bool, default=False)
    parser.add_argument('--sp_num', type=int, default=5)
    parser.add_argument("--seed_specify", action="store_true",
                        default=False, help="Random or specify seed for numpy/torch")
    parser.add_argument("--runing_id", type=int,
                        default=1, help="the runing index of experiment")

    parser.add_argument("--weighted_clip_step", type=int, default=40000)
    parser.add_argument("--weighted_clip_init", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--weighted_clip', type=bool, default=False)
    parser.add_argument('--all_state_clip', type=bool, default=False,
                        help="Which group to run on")
    parser.add_argument('--true_rho_s', type=bool, default=False,
                        help="Which group to run on")
    parser.add_argument('--group_name', type=str, default='test',
                        help="Which group to run on")
    parser.add_argument('--dynamic_l3', type=bool, default=False)
    parser.add_argument('--dcmode2_save', type=bool, default=False)
    parser.add_argument('--L3_coeff', type=float, default=1e-5)
    parser.add_argument('--dcmode', type=int, default=1)
    parser.add_argument('--dc_check_output', type=bool, default=False)
    parser.add_argument('--delta_reset', type=bool, default=False)
    parser.add_argument('--delta_decay', type=bool, default=False)
    parser.add_argument('--min_clip_params', type=float, default=0.1)
    parser.add_argument('--solve_check', type=bool, default=False)
    parser.add_argument('--linear_search_check', type=bool, default=False)
    parser.add_argument('--overflow_save', type=bool, default=False)
    parser.add_argument('--clip_delta_max_eps', type=float, default=0.3)
    parser.add_argument('--clip_delta_min_eps', type=float, default=1e-2)
    parser.add_argument('--clip_delta_smooth', type=bool, default=False)
    parser.add_argument('--joint_optim', type=bool, default=False)
    parser.add_argument('--multi_rollout', type=bool, default=True)
    parser.add_argument('--use_q', type=bool, default=False)
    parser.add_argument('--lucky_guy', type=bool, default=False)
    parser.add_argument('--save_not_solved', type=bool, default=False)
    parser.add_argument('--solve_eps', type=float, default=1e-6)
    parser.add_argument('--tol_iteration', type=int, default=100)
    parser.add_argument('--clip_update_num', type=int, default=1)
    parser.add_argument('--solve_max_iter', type=int, default=2000)
    parser.add_argument('--clip_lr_decay', type=float, default=0.9)
    parser.add_argument('--clip_param_lr', type=float, default=0.1)
    parser.add_argument('--dynamic_clip_tag', type=bool, default=False)

    parser.add_argument("--idv_para", action='store_false', default=True)
    parser.add_argument('--optim_reset', type=bool, default=False)
    parser.add_argument('--matrix_test', type=bool, default=False)
    parser.add_argument('--aga_F', type=float, default=1)
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--target_dec', type=bool, default=False)
    parser.add_argument('--soft_target', type=bool, default=False)
    parser.add_argument('--new_period', type=bool, default=False)

    parser.add_argument('--aga_first', type=bool, default=False)
    parser.add_argument('--aga_tag', type=bool, default=False)


    return parser
