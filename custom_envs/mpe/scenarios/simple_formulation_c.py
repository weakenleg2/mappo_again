# noqa: D212, D415
"""
# Simple Adversary

```{figure} mpe_simple_adversary.gif
:width: 140px
:name: simple_adversary
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_adversary_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete/Continuous                              |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [adversary_0, agent_0,agent_1]`         |
| Agents             | 3                                                |
| Action Shape       | (5)                                              |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5))                   |
| Observation Shape  | (8),(10)                                         |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


In this environment, there is 1 adversary (red), N good agents (green), N landmarks (default N=2). All agents observe the position of landmarks and other agents. One landmark is the 'target landmark' (colored green). Good agents are rewarded based on how close the closest one of them is to the
target landmark, but negatively rewarded based on how close the adversary is to the target landmark. The adversary is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark. All rewards are unscaled Euclidean distance (see main MPE documentation for
average distance). This means good agents have to learn to 'split up' and cover all landmarks to deceive the adversary.

Agent observation space: `[goal_rel_position, landmark_rel_position, other_agent_rel_positions]`

Adversary observation space: `[landmark_rel_position, other_agents_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
```



`N`:  number of good agents and landmarks

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from custom_envs.mpe.core import Agent, Landmark, World
from custom_envs.mpe.scenario import BaseScenario
from custom_envs.mpe.simple_env import SimpleEnv, make_env

from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        penalty_ratio = 0.5,
        full_comm=True,
        local_ratio=0.5,
        max_cycles=25,
        delay = 10,
        packet_drop_prob=0.1,
        continuous_actions=False,
        render_mode=None,
        ):
        EzPickle.__init__(
            self, N=N, 
            penalty_ratio=penalty_ratio,  
            local_ratio=local_ratio, 
            full_comm=full_comm,
            max_cycles=max_cycles, 
            delay = delay,packet_drop_prob=packet_drop_prob,

            continuous_actions=continuous_actions, 
            render_mode=render_mode
        )
        scenario = Scenario()
        world = scenario.make_world(N, penalty_ratio, full_comm, delay, packet_drop_prob)
        super().__init__(
            # self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_formulation_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


from scipy.optimize import linear_sum_assignment
collision_penal = 0
vision = 10
class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        # self.current_time = 0
        self.message_queue = []
    def action_callback(self, agent, current_step, _): 
      #To test full comm
      if self.full_comm:
        agent.action.c = np.array([1, 0])

      if agent.action.c[0] > agent.action.c[1] and np.random.rand() >= self.packet_drop_prob:
        scheduled_time = current_step + self.delay
        message_content = np.concatenate((agent.state.c, [0]))
        self.message_queue.append((scheduled_time, agent.name, message_content))
        

        agent.color = np.array([0, 1, 0])
      else:
        agent.color = np.array([0.35, 0.35, 0.85])
        self.last_message[agent.name][-1] += 1
        
      
      return agent.action

    def make_world(self, N=3, penalty_ratio=0.5, full_comm=False,delay = 2, packet_drop_prob=0.2):
        world = World()
        # set any world properties first
        world.dim_c = 2

        self.collision_penal = collision_penal
        self.vision = vision
        num_agents = N
        num_landmarks = 2
        self.n_agents = N
        # self.n_collisions = 0
        world.collaborative = True
        self.full_comm = full_comm
        self.penalty_ratio = penalty_ratio 
        self.last_message = {}
        self.packet_drop_prob = packet_drop_prob
        self.delay = delay



        self.total_sep = 1.25
        self.arena_size = 1
        self.ideal_sep = self.total_sep / (self.n_agents - 1)
        self.rewards = np.zeros(self.n_agents)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = False
            agent.size = 0.03
            agent.action_callback = self.action_callback
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
        # make initial conditions
        self.reset_world(world)

        world.dists = []
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = np.random.uniform(-0.25, +0.25, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        theta = np.random.uniform(0, 2 * np.pi)
        loc = world.landmarks[0].state.p_pos + self.total_sep * np.array([np.cos(theta), np.sin(theta)])
        # find a suitable theta such that landmark 1 is within the bounds
        while not (abs(loc[0]) < self.arena_size and abs(loc[1]) < self.arena_size):
            theta += np.radians(5)
            loc = world.landmarks[0].state.p_pos + self.total_sep * np.array([np.cos(theta), np.sin(theta)])

        world.landmarks[1].state.p_pos = loc
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

        self.expected_positions = [
            world.landmarks[0].state.p_pos + i * self.ideal_sep * np.array([np.cos(theta), np.sin(theta)])
            for i in range(len(world.agents))]

        world.steps = 0
        world.dists = []



    def is_obs(self,entity1,entity2):
        delt_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delt_pos)))
        return True if dist < self.vision else False
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world, global_reward=None):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        if agent.name == 'agent_0':
            rew = 0
            world.dists = np.array([[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_positions]
                                    for a in world.agents])
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.min_dists = self._bipartite_min_dists(world.dists)
            # the reward is normalized by the number of agents
            rew = -np.mean(self.min_dists)

            collision_rew = 0
            for b in world.agents:
                for a in world.agents:
                    if self.is_collision(a, b):
                        collision_rew -= self.collision_penal
            collision_rew /= (2 * self.n_agents)
            rew += collision_rew
            if agent.action.c[0] > agent.action.c[1]:
            # print(self.penalty_ratio)
                rew -= self.penalty_ratio

            rew = np.clip(rew, -15, 15)
            self.rewards = np.full(self.n_agents, rew)
            world.min_dists = self.min_dists
        # print("self.rewards.mean()",self.rewards.mean())
        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists
    def process_messages(self, current_step):
        for i in range(len(self.message_queue) - 1, -1, -1):
            scheduled_time, name, message = self.message_queue[i]
            if scheduled_time == current_step:
                self.last_message[name] = message
                self.message_queue.pop(i)
    def global_reward(self, world):
        rew = 0.0
        # for lm in world.landmarks:
        #     dists = [
        #         np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
        #         for a in world.agents
        #     ]
        #     rew -= min(dists)
        return rew

    def observation(self, agent, world,current_step):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if self.is_obs(agent,entity):
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(np.zeros_like(entity.state.p_pos - agent.state.p_pos))
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        if current_step is not None:
            self.process_messages(current_step)
        for other in world.agents:
            if other is agent: continue
            message = self.last_message[other.name]
            comm.append(message)
            # comm.append(other.state.c)
            if self.is_obs(agent,other):
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append(np.zeros_like(other.state.p_pos - agent.state.p_pos))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    