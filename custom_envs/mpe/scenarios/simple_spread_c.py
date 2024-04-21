# noqa
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v2` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from custom_envs.mpe.core import Agent, Landmark, World
from custom_envs.mpe.scenario import BaseScenario
from custom_envs.mpe.simple_env import SimpleEnv, make_env

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
            self, N=N, penalty_ratio=penalty_ratio,  
            local_ratio=local_ratio, full_comm=full_comm,
            delay = delay,packet_drop_prob=packet_drop_prob,
            max_cycles=max_cycles, continuous_actions=continuous_actions, 
            render_mode=render_mode
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N, penalty_ratio, full_comm, delay, packet_drop_prob)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_v2"

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

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
        message_content = np.concatenate((agent.state.p_pos, [0]))
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
        num_agents = N
        num_landmarks = N
        self.n_collisions = 0
        world.collaborative = True
        self.full_comm = full_comm
        self.penalty_ratio = penalty_ratio 
        self.last_message = {}
        self.world_min = -1 - (0.1 * num_agents)
        self.world_max = 1 + (0.1 * num_agents)
        self.packet_drop_prob = packet_drop_prob
        self.delay = delay

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = False
            agent.size = 0.15
            agent.action_callback = self.action_callback
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world):
        # random properties for agents
        self.n_collisions = 0
        for _, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # random properties for landmarks
        for _, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(self.world_min, self.world_max, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for _, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(self.world_min, self.world_max, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world, global_reward=None):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if a.name == agent.name:
                    continue
                is_collision = (self.is_collision(a, agent))
                if is_collision:
                    self.n_collisions += 1
                rew -= 1.0 * is_collision

        #Add penalty for communication
        if global_reward and agent.action.c[0] > agent.action.c[1]:
        # if agent.action.c[0] > agent.action.c[1]:
          rew -= self.penalty_ratio
        return rew

    def global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew
    def process_messages(self, current_step):
        for i in range(len(self.message_queue) - 1, -1, -1):
            scheduled_time, name, message = self.message_queue[i]
            if scheduled_time == current_step:
                self.last_message[name] = message
                self.message_queue.pop(i)

    def observation(self, agent, world,current_step):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)
        # communication of all other agents
        entity_pos = np.concatenate(entity_pos)
        comm = []
        if current_step is not None:
            self.process_messages(current_step)
        for other in world.agents:
            #com_flag = 0
            if other is agent:
                continue
            message = self.last_message[other.name]
            comm.append(message)

        comm = np.concatenate(comm)
        # print("comm",comm)
        obs = np.concatenate(
            (agent.state.p_pos, agent.state.p_vel,
            entity_pos, comm))
        # print("obs",obs)
        # agent still gets the landmarks(world info)

        return obs
