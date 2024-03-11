from ray import air
from ray.tune.schedulers.pb2 import PB2
import argparse
from ray import tune
from ray.tune.schedulers.pb2 import PB2
from train_mappo import simple_train

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('num_agents', type=int, default=2)
  parser.add_argument('--logdir', type=str, default="/ray_results")
  parser.add_argument('--full_com', action='store_true')
  return parser.parse_args()

best_rewards = {}
iterations_since_best = {}

def stopper(trial_id, result):
    key = str(trial_id)
    global best_rewards, iterations_since_best

    if not key in best_rewards:
       best_rewards[key] = -float('inf')
       iterations_since_best[key] = 0

    improvement_threshold = 0.1
    patience = 100

    if result["average_episode_rewards"] > best_rewards[key] + improvement_threshold:
        best_rewards[key] = result["average_episode_rewards"]
        iterations_since_best[key] = 0
    else:
        iterations_since_best[key] += 1

    if iterations_since_best[key] >= patience:
        return True
    else:
       return False


if __name__ == '__main__':
  args = parse_args()
  config = {
      "algorithm_name": tune.choice(["mappo", 'rmappo']),
      "pop_art": tune.choice([True, False]),
      "use_ReLU": tune.choice([True, False]),
      "episode_length": 25,
      "env_name": "MPE",
      "scenario_name": "simple_spread",
      "experiment_name": "mappo_search_" + str(args.num_agents) + "_agents" + ("_full_com" if args.full_com else "_limited_com"),
      "cuda": False,
      "n_training_threads": 1,
      "n_rollout_threads": 1,
      "n_eval_rollout_threads": 1,
      "n_render_rollout_threads": 1,
      "num_env_steps": 1e7,
      "num_agents": args.num_agents,
      "share_policy": True,
      "use_centralized_V": True,
      "local_ratio": 0.5,
      "full_comm": args.full_com,
  }
  pb2_config = {
      "n_trajectories": [1, 5000],
      "actor_hidden_size": [32, 2048],
      "critic_hidden_size": [32, 2048],
      "layer_N": [1, 3],
      "critic_lr": [1e-7, 1e-4],
      "ppo_epoch": [1, 20],
      "clip_param": [0.1, 0.5],
      "gae_lambda": [0.9, 1],
      "gamma": [0.9, 1],
      "lr": [1e-7, 1e-3],
      "entropy_coef": [0.001, 0.1],
      "comm_penalty": [0.001, 1.0],
  }

  pbt = PB2(
      time_attr="training_iteration",
      perturbation_interval=5,
      hyperparam_bounds=pb2_config,
  )

  analysis = tune.run(
      simple_train,
      config=config,
      scheduler=pbt,
      num_samples=10,
      stop=stopper,
      name="pbt",
      local_dir=args.logdir,
      metric="average_episode_rewards",
      mode="max",
  )
