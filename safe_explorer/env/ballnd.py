import gym
from gym.spaces import Box, Dict
import numpy as np
from numpy import linalg as LA

import math

from safe_explorer.core.config import Config

class BallND(gym.Env):
    def __init__(self):
        # get config
        config = Config.get().env.ballnd
        # set attributes
        self.n = config.n
        self.target_margin = config.target_margin
        self.agent_slack = config.agent_slack
        self.episode_length = config.episode_length
        self.time_step = config.time_step
        self.respawn_interval = config.respawn_interval
        self.target_noise_std = config.target_noise_std
        self.enable_reward_shaping = config.enable_reward_shaping
        self.reward_shaping_slack = config.reward_shaping_slack
        # Set the properties for spaces
        self.action_space = Box(low=-math.inf, high=math.inf, shape=(self.n,), dtype=np.double)
        self.observation_space = Box(low=0, high=1, shape=(3*self.n,), dtype=np.double) # ['ball position','ball velocity','target position']
        # rendering viewer
        self.viewer = None

        # Sets all the episode specific variables
        self.reset()
        
    def reset(self, random_agent_position=False):
        if random_agent_position:
            self.ball_pos = np.random.random(self.n)
        else:
            self.ball_pos = 0.5 * np.ones(self.n, dtype=np.float32)
        self.ball_velocity = np.zeros(self.n, dtype=np.float32)
        self._reset_target_location()
        self.time = 0.
        self.target_respawn_time = 0.
        return np.concatenate([self.ball_pos, self.ball_velocity, self._get_noisy_target_position()])

    def _is_ball_outside_boundary(self):
        return np.any(self.ball_pos < 0) or np.any(self.ball_pos > 1)
    
    def _is_ball_outside_shaping_boundary(self):
        return np.any(self.ball_pos < self.reward_shaping_slack) \
               or np.any(self.ball_pos > 1 - self.reward_shaping_slack)

    def _get_reward(self):
        if self.enable_reward_shaping and self._is_ball_outside_shaping_boundary():
            return -1
        else:
            return np.clip(1 - 10 * LA.norm(self.ball_pos - self._target_position)**2, 0, 1)
    
    def _reset_target_location(self):
        self._target_position = \
            (1 - 2 * self.target_margin) * np.random.random(self.n) + self.target_margin

    def _move_ball(self):
        # advance time
        self.time += self.time_step
        self.target_respawn_time += self.time_step
        # move ball
        self.ball_pos += self.time_step * self.ball_velocity
        self.ball_velocity *= 0.99 # dampening
        # Check if the target needs to be relocated
        if self.target_respawn_time > self.respawn_interval:
            self._reset_target_location()
            self.target_respawn_time = 0.
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self.target_noise_std, self.n)
    
    def get_num_constraints(self):
        return 2*self.n

    def get_constraint_values(self):
        # define all constraint bounds as 0: C_i = 0 for all i
        # set lower and upper constraint for each dimension: slack < ball position < 1 - slack
        # slack < ball position --> slack - ball position < 0
        min_constraints = self.agent_slack - self.ball_pos
        # ball position < 1 - slack --> ball position - (1 - slack) < 0
        max_constraint = self.ball_pos - (1 - self.agent_slack)
        
        return np.concatenate([min_constraints, max_constraint])

    def step(self, action):
        # set action
        self.ball_velocity = action
        # execute 4 times steps; move ball 4 time steps
        for _ in range(0, 4):
            self._move_ball()
        # get reward         
        reward = self._get_reward()
        # construct new state
        state = np.concatenate([self.ball_pos, self.ball_velocity, self._get_noisy_target_position()])
        # check for constraint violation
        constraint_violation = self._is_ball_outside_boundary()
        # check if done: (i) constraint violation or (ii) reached max episode length 
        done = constraint_violation or int(self.time // 1) > self.episode_length

        return state, reward, done, {'constraint_violation': constraint_violation}

    def render(self, mode="human"):
        # only implemented render ball-1D domain for debugging purposes
        if self.n != 1:
            return None
        
        screen_width = 600
        screen_height = 400
        screen_padding = 50

        world_width = 1.
        scale = (screen_width - 2*screen_padding) / world_width

        ball_dia = 0.025
        y = screen_height / 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            cube = rendering.Line(start=(screen_padding,y), end=(screen_padding + 1.*scale,y))
            cube.set_color(0,0,0)
            self.viewer.add_geom(cube)

            self.target = rendering.make_circle(ball_dia * scale)
            self.target.set_color(0.8,0.,0.)
            self.targettrans = rendering.Transform()
            self.target.add_attr(self.targettrans)
            self.viewer.add_geom(self.target)

            self.ball = rendering.make_circle(ball_dia * scale)
            self.ball.set_color(0.,0.8,0.)
            self.balltrans = rendering.Transform()
            self.ball.add_attr(self.balltrans)
            self.viewer.add_geom(self.ball)

        self.balltrans.set_translation(screen_padding + self.ball_pos[0]*scale, y)
        self.targettrans.set_translation(screen_padding + self._target_position[0]*scale, y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None