from pettingzoo.sisl import multiwalker_v9

env = multiwalker_v9.env()
env.reset(seed=42)
print(env.observation_space)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()