import copy
import math

import Box2D
import numpy as np
import pygame
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
from gymnasium import spaces
from gymnasium.utils import seeding
from pygame import gfxdraw
from custom_envs.multiwalker_communicate.multiwalker_c._utils import Agent

MAX_AGENTS = 40

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

PACKAGE_POLY = [(-120, 5), (120, 5), (120, -5), (-120, -5)]

PACKAGE_LENGTH = 240

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

WALKER_SEPERATION = 10  # in steps


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # if walkers fall on ground
        for i, walker in enumerate(self.env.walkers):
            if walker.hull is not None:
                if walker.hull == contact.fixtureA.body:
                    if self.env.package != contact.fixtureB.body:
                        self.env.fallen_walkers[i] = True
                if walker.hull == contact.fixtureB.body:
                    if self.env.package != contact.fixtureA.body:
                        self.env.fallen_walkers[i] = True

        # if package is on the ground
        if self.env.package == contact.fixtureA.body:
            if contact.fixtureB.body not in [w.hull for w in self.env.walkers]:
                self.env.game_over = True
        if self.env.package == contact.fixtureB.body:
            if contact.fixtureA.body not in [w.hull for w in self.env.walkers]:
                self.env.game_over = True

        # self.env.game_over = True
        for walker in self.env.walkers:
            if walker.hull is not None:
                for leg in [walker.legs[1], walker.legs[3]]:
                    if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                        leg.ground_contact = True

    def EndContact(self, contact):
        for walker in self.env.walkers:
            if walker.hull is not None:
                for leg in [walker.legs[1], walker.legs[3]]:
                    if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                        leg.ground_contact = False


class BipedalWalker(Agent):
    def __init__(
        self,
        world,
        full_comm,
        
        init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
        init_y=TERRAIN_HEIGHT + 2 * LEG_H,
        n_walkers=2,
        seed=None,
    ):
        self.world = world
        self._n_walkers = n_walkers
        self.hull = None
        self.init_x = init_x
        self.init_y = init_y
        self.walker_id = -int(self.init_x)
        self._seed(seed)
        self.comm_signal = 0
        self.full_comm = full_comm
        self.communication_count = 0
        
    def _destroy(self):
        if not self.hull:
            return
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self._destroy()
        init_x = self.init_x
        init_y = self.init_y
        self.comm_signal = 0
        # self.walker_message = np.zeros(11)
        self.communication_count = 0
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]
                ),
                density=5.0,
                friction=0.1,
                groupIndex=self.walker_id,
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
                    density=1.0,
                    restitution=0.0,
                    groupIndex=self.walker_id,
                ),  # collide with ground only
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
                    density=1.0,
                    restitution=0.0,
                    groupIndex=self.walker_id,
                ),
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]

    def apply_action(self, action):
        self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
        self.joints[0].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
        )
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
        self.joints[1].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
        )
        self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
        self.joints[2].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
        )
        self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
        self.joints[3].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
        )
        if self.full_comm:
            # random value, means random skip
            # naive
            self.comm_signal = 1
        else:
            self.comm_signal = action[4]
    def get_observation(self):
        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
        # only position info, maybe including ground

        state = [
            # Normal angles up to 0.5 here, but sure more is possible.
            self.hull.angle,
            2.0 * self.hull.angularVelocity / FPS,
            # Normalized to get -1..1 range
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].angle,
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        # if self.comm_signal>0:
        #         self.walker_message = np.concatenate(([l_dis.fraction for l_dis in self.lidar],[0]))
        #     # self.last_message = neighbor_obs

        # else:
        #     self.walker_message[-1]+=1
        # obs.append(np.concatenate([walker_obs, self.last_message]))
        # state = np.concatenate([state,self.walker_message]) 
        state += [l_dis.fraction for l_dis in self.lidar]
        assert len(state) == 24

        return state

    @property
    def observation_space(self):
        # 24 original obs (joints, etc), 2 displacement obs for each neighboring walker, 3 for package, 1 for timer
        return spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(24 + 4 + 3 + 1,),
            dtype=np.float32,
        )
    @property
    def state_space(self):
        # Define the state space
        # This is just an example. Modify according to your environment's state properties
        return spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(25 + 4 + 3 + 1,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        # return spaces.Box(
        #     low=np.float32(-1), high=np.float32(1), shape=(4,), dtype=np.float32
        # )
        return spaces.Tuple([
                spaces.Box(
            low=np.float32(-1), high=np.float32(1), shape=(4,), dtype=np.float32
        ),
                spaces.Discrete(2),
                ]
            )
    


class MultiWalkerEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    hardcore = False

    def __init__(
        self,
        n_walkers=3,
        position_noise=1e-3,
        angle_noise=1e-3,
        forward_reward=1.0,
        terminate_reward=-100.0,
        fall_reward=-10.0,
        shared_reward=False,
        terminate_on_fall=True,
        remove_on_fall=True,
        terrain_length=TERRAIN_LENGTH,
        penalty_ratio = 0.5,
        full_comm=True,
        delay = 10,
        packet_drop_prob=0.1,
        bandwidth_limit = 10,
        max_cycles=500,
        render_mode=None,
    ):
        """Initializes the `MultiWalkerEnv` class.

        n_walkers: number of bipedal walkers in environment
        position_noise: noise applied to agent positional sensor observations
        angle_noise: noise applied to agent rotational sensor observations
        forward_reward: reward applied for an agent standing, scaled by agent's x coordinate
        fall_reward: reward applied when an agent falls down
        shared_reward: whether reward is distributed among all agents or allocated locally
        terminate_reward: reward applied for each fallen walker in environment
        terminate_on_fall: toggles whether agent is done if it falls down
        terrain_length: length of terrain in number of steps
        max_cycles: after max_cycles steps all agents will return done
        """
        self.n_walkers = n_walkers
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self.forward_reward = forward_reward
        self.fall_reward = fall_reward
        self.terminate_reward = terminate_reward
        self.terminate_on_fall = terminate_on_fall
        # print(self.terminate_on_fall)
        self.local_ratio = 1 - shared_reward
        self.remove_on_fall = remove_on_fall
        self.terrain_length = terrain_length
        self.seed_val = None
        self.full_comm = full_comm
        # print(self.full_comm)
        self.penalty_ratio = penalty_ratio
        self.seed()
        self.setup()
        self.screen = None
        self.isopen = True
        self.agent_list = list(range(self.n_walkers))
        self.last_rewards = [0 for _ in range(self.n_walkers)]
        self.last_dones = [False for _ in range(self.n_walkers)]
        self.last_obs = [None for _ in range(self.n_walkers)]
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.frames = 0
        self.delay = delay
        self.packet_drop_prob = packet_drop_prob
        self.bandwidth_limit = bandwidth_limit
        self.current_round_message_count = 0

        # print(self.delay)
        
        self.last_message = np.zeros(8)
        self.message_buffer = []
        
    def send_message(self, message, target_agent_id, delay):
        self.current_round_message_count += 1
        
        if np.random.rand() >= self.packet_drop_prob:

            current_step = self.get_cycle_count()
            deliver_at_step = current_step + delay
            self.message_buffer.append((message, deliver_at_step, target_agent_id))
            
        if len(self.message_buffer) > (delay+5):
            self.message_buffer.pop(0)

    
    def process_message_buffer(self, agent_id, current_step):
        received_message = np.zeros(8)
        new_buffer = []
        for message, deliver_at_step, target_agent_id in self.message_buffer:
            if deliver_at_step <= current_step and target_agent_id == agent_id:
                received_message = message  # Last message for this agent in time window
            else:
                new_buffer.append((message, deliver_at_step, target_agent_id))
        self.message_buffer = new_buffer
        return received_message

    def get_cycle_count(self):
        return self.frames
    def get_param_values(self):
        return self.__dict__

    def setup(self):
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.start_x = [
            init_x + WALKER_SEPERATION * i * TERRAIN_STEP for i in range(self.n_walkers)
        ]
        self.walkers = [
            BipedalWalker(self.world,self.full_comm,init_x=sx, init_y=init_y, seed=self.seed_val)
            for sx in self.start_x
        ]
        self.num_agents = len(self.walkers)
        self.observation_space = [agent.observation_space for agent in self.walkers]
        self.action_space = [agent.action_space for agent in self.walkers]
        self.state_space = [agent.state_space for agent in self.walkers]

        self.package_scale = self.n_walkers / 1.75
        self.package_length = PACKAGE_LENGTH / SCALE * self.package_scale

        self.total_agents = self.n_walkers

        self.prev_shaping = np.zeros(self.n_walkers)
        self.prev_package_shaping = 0.0

    @property
    def agents(self):
        return self.walkers

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        self.seed_val = seed_
        for walker in getattr(self, "walkers", []):
            walker._seed(seed_)
        return [seed_]

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.package)
        self.package = None

        for walker in self.walkers:
            walker._destroy()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
    def get_communication_counts(self):
        # Returns a list or dictionary of communication counts for each agent
        # Example implementation (you would need to implement the logic to track counts)
        # print([walker.communication_count for walker in self.walkers])
        return [walker.communication_count for walker in self.walkers]

    def reset(self):
        self.setup()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.fallen_walkers = np.zeros(self.n_walkers, dtype=bool)
        self.prev_shaping = np.zeros(self.n_walkers)
        self.prev_package_shaping = 0.0
        self.scroll = 0.0
        self.lidar_render = 0
        self.last_message = np.zeros(8)
        self.message_buffer = []
        self.current_round_message_count = 0
        

        self._generate_package()
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        self.drawlist = copy.copy(self.terrain)

        self.drawlist += [self.package]

        for walker in self.walkers:
            walker._reset()
            self.drawlist += walker.legs
            self.drawlist += [walker.hull]
        r, d, o = self.scroll_subroutine()
        self.last_rewards = [0 for _ in range(self.n_walkers)]
        self.last_dones = [False for _ in range(self.n_walkers)]
        for i in range(self.n_walkers):
            # message = self.process_message_buffer(i,current_step = current_step)
            # print("message",message)
            # self.last_obs[i] = np.concatenate([mod_obs[i], message])
            self.last_obs[i] = np.concatenate([o[i],np.zeros(8)])
        self.frames = 0

        return self.observe(0)

    def scroll_subroutine(self):
        xpos = np.zeros(self.n_walkers)
        obs = []
        done = False
        rewards = np.zeros(self.n_walkers)

        for i in range(self.n_walkers):
            if self.walkers[i].hull is None:
                obs.append(np.zeros_like(self.observation_space[i].low))
                continue
            pos = self.walkers[i].hull.position
            x, y = pos.x, pos.y
            xpos[i] = x

            walker_obs = self.walkers[i].get_observation()
            # ###we can consider observation of lidar here
            neighbor_obs = []
            for j in [i - 1, i + 1]:
                # if no neighbor (for edge walkers)
                if j < 0 or j == self.n_walkers or self.walkers[j].hull is None:
                    neighbor_obs.append(0.0)
                    neighbor_obs.append(0.0)
                else:
                    xm = (self.walkers[j].hull.position.x - x) / self.package_length
                    ym = (self.walkers[j].hull.position.y - y) / self.package_length
                    neighbor_obs.append(self.np_random.normal(xm, self.position_noise))
                    neighbor_obs.append(self.np_random.normal(ym, self.position_noise))
            xd = (self.package.position.x - x) / self.package_length
            yd = (self.package.position.y - y) / self.package_length
            neighbor_obs.append(self.np_random.normal(xd, self.position_noise))
            neighbor_obs.append(self.np_random.normal(yd, self.position_noise))
            neighbor_obs.append(
                self.np_random.normal(self.package.angle, self.angle_noise)
            )
            assert len(self.last_message) == (len(neighbor_obs)+1)
            if self.walkers[i].comm_signal>0:
                self.last_message = np.concatenate((neighbor_obs,[0]))
            # self.last_message = neighbor_obs

            else:
                self.last_message[-1]+=1
            self.send_message(self.last_message,i,delay=self.delay)
            
            # obs.append(np.concatenate([walker_obs, self.last_message]))
            obs.append(walker_obs)

            shaping = -5.0 * abs(walker_obs[0])

            rewards[i] = shaping - self.prev_shaping[i]
            self.prev_shaping[i] = shaping

        package_shaping = self.forward_reward * 130 * self.package.position.x / SCALE
        rewards += package_shaping - self.prev_package_shaping
        self.prev_package_shaping = package_shaping

        self.scroll = (
            xpos.mean()
            - VIEWPORT_W / SCALE / 5
            - (self.n_walkers - 1) * WALKER_SEPERATION * TERRAIN_STEP
        )

        done = [False] * self.n_walkers
        for i, (fallen, walker) in enumerate(zip(self.fallen_walkers, self.walkers)):
            if fallen:
                rewards[i] += self.fall_reward
                if self.remove_on_fall:
                    walker._destroy()
                if not self.terminate_on_fall:
                    rewards[i] += self.terminate_reward
                done[i] = True
        if (
            (self.terminate_on_fall and np.sum(self.fallen_walkers) > 0)
            or self.game_over
            or self.package.position.x < 0
        ):
            rewards += self.terminate_reward
            done = [True] * self.n_walkers
        elif (
            self.package.position.x
            > (self.terrain_length - TERRAIN_GRASS) * TERRAIN_STEP
        ):
            done = [True] * self.n_walkers

        return rewards, done, obs

    def step(self, action, agent_id, is_last):
        # action is array of size 4
        # action = action.reshape(4)
        assert self.walkers[agent_id].hull is not None, agent_id
        self.walkers[agent_id].apply_action(action)
        if is_last:
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
            current_step = self.get_cycle_count()
            rewards, done, mod_obs = self.scroll_subroutine()
            
            # print("mode",mod_obs)

            for i in range(self.n_walkers):
                if i >= self.current_round_message_count:
                    message = np.zeros(8)
                else:
                    message = self.process_message_buffer(i,current_step = current_step)
                # print("message",message)
                self.last_obs[i] = np.concatenate([mod_obs[i], message])
            for i in range(self.n_walkers):
                # print(self.walkers[i].comm_signal)
                if self.walkers[i].comm_signal>0:
                    # paramter tuning
                    # if rewards[i]<0:
                    # print(rewards[i])
                    rewards[i] = rewards[i]-self.penalty_ratio
                    # need to be tuned
                    # else:
                    #     rewards[i] = rewards[i]*(1-self.penalty_ratio)
                    self.walkers[i].communication_count += 1
            self.current_round_message_count = 0     
            global_reward = rewards.mean()
            local_reward = rewards * self.local_ratio
            self.last_rewards = (
                global_reward * (1.0 - self.local_ratio)
                + local_reward * self.local_ratio
            )
            # the reward is always global
            self.last_dones = done
            self.frames = self.frames + 1

        if self.render_mode == "human":
            self.render()

    def get_last_rewards(self):
        return dict(
            zip(
                list(range(self.n_walkers)),
                map(lambda r: np.float64(r), self.last_rewards),
            )
        )

    def get_last_dones(self):
        return dict(zip(self.agent_list, self.last_dones))

    def get_last_obs(self):
        return dict(
            zip(
                list(range(self.n_walkers)),
                [walker.get_observation() for walker in self.walkers],
            )
        )

    def observe(self, agent):
        o = self.last_obs[agent]
        o = np.array(o, dtype=np.float32)
        return o