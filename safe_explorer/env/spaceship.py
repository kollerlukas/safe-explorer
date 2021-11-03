import gym
from gym.spaces import Box
import numpy as np

import math

from safe_explorer.core.config import Config
from safe_explorer.env.constraintenv import ConstraintEnv


class Spaceship(ConstraintEnv):
    def __init__(self):
        # get config
        config = Config.get().env.spaceship
        # set attributes
        self.arena = config.arena
        self.agent_slack = config.agent_slack
        self.episode_length = config.episode_length
        self.time_step = config.time_step
        self.enable_reward_shaping = config.enable_reward_shaping
        self.reward_shaping_margin = config.reward_shaping_margin
        # action space: a \in [-1,1]
        self.action_space = Box(low=-1., high=1., shape=(2,), dtype=np.double)
        # state space: ['spaceship's location','spaceship's velocity']
        self.observation_space = Box(
            low=-math.inf, high=math.inf, shape=(2+2,), dtype=np.double)
        # rendering viewer
        self.viewer = None

        # Sets all the episode specific variables
        self.reset()

        # set target location
        if self.arena:
            self.target_pos = np.array([0.8, 0.8])
        else:
            self.target_pos = np.array([0.5, 1.])

    def reset(self):
        if self.arena:
            # spaceship start randomly in lower third: agent_slack <= x <= 0.3-agent_slack, agent_slack <= y <= 0.3-agent_slack
            self.spaceship_pos = (0.3 - 2*self.agent_slack) * \
                np.random.random(2) + self.agent_slack
        else:
            # spaceship start randomly in lower third: agent_slack <= x <= 1-agent_slack, 0 <= y <= 0.3
            self.spaceship_pos = np.multiply(np.array(
                [1. - 2*self.agent_slack, 0.3]), np.random.random(2)) + np.array([self.agent_slack, 0.])
        self.spaceship_velocity = np.array([0., 0.])
        self.time = 0.
        return np.concatenate([self.spaceship_pos, self.spaceship_velocity])

    def _is_spaceship_out_of_bounds(self):
        if self.arena:
            return np.any(self.spaceship_pos < 0.) or np.any(1. < self.spaceship_pos)
        else:
            return self.spaceship_pos[0] < 0. or 1. < self.spaceship_pos[0]

    def _is_target_reached(self):
        return np.linalg.norm(self.spaceship_pos - self.target_pos) < 0.025

    def _get_reward(self):
        if self._is_target_reached():
            return 1000
        elif self.enable_reward_shaping:
            if self.arena:
                if (np.any(self.spaceship_pos <= self.reward_shaping_margin)
                        or np.any((1.-self.reward_shaping_margin) <= self.spaceship_pos)):
                    return -1000
                else:
                    return 0
            else:
                if (self.spaceship_pos[0] <= self.reward_shaping_margin
                        or (1.-self.reward_shaping_margin) <= self.spaceship_pos[0]):
                    return -1000
                else:
                    return 0
        else:
            return 0

    def _move_spaceship(self, thrusters):
        # advance time
        self.time += self.time_step
        # adjust spaceship's velocity
        self.spaceship_velocity += self.time_step * thrusters
        # move spaceship
        self.spaceship_pos += self.time_step * self.spaceship_velocity
        self.spaceship_velocity *= 0.95  # dampening

    def get_num_constraints(self):
        if self.arena:
            return 4
        else:
            return 2

    def get_constraint_values(self):
        # define all constraint bounds as 0: C_i = 0 for all i
        if self.arena:
            # set left and right constraint for the x- & y-dimension:
            #     slack < spaceship_pos < 1 - slack
            # slack < spaceship_pos --> slack - spaceship_pos < 0
            left_constraints = self.agent_slack - self.spaceship_pos
            # spaceship_pos < 1 - slack --> spaceship_pos - (1 - slack) < 0
            right_constraints = self.spaceship_pos - (1 - self.agent_slack)

            return np.concatenate([left_constraints, right_constraints])
        else:
            # set left and right constraint for the y-dimension:
            #     slack < spaceship_pos[0] < 1 - slack
            # slack < spaceship_pos[0] --> slack - spaceship_pos[0] < 0
            left_constraint = self.agent_slack - self.spaceship_pos[0]
            # spaceship_pos[0] < 1 - slack --> spaceship_pos[0] - (1 - slack) < 0
            right_constraint = self.spaceship_pos[0] - (1 - self.agent_slack)

            return np.array([left_constraint, right_constraint])

    def step(self, action):
        # move the spaceship with action
        self._move_spaceship(action)
        # get reward
        reward = self._get_reward()
        # construct new state
        state = np.concatenate([self.spaceship_pos, self.spaceship_velocity])
        # check for constraint violation: hit the left or right wall
        constraint_violation = self._is_spaceship_out_of_bounds()
        # check if done: (i) constraint violation or (ii) target is reached or (iii) reached max episode length
        done = constraint_violation \
            or self._is_target_reached() \
            or self.time > self.episode_length

        return state, reward, done, {'constraint_violation': constraint_violation}

    def render(self, mode="human"):
        # only implemented render ball-1D & -3D domain for debugging purposes

        # set screen dimensions
        screen_width, screen_height = 500, 800
        if self.arena:
            screen_height = 500
        # set padding from edges of the screen
        padx, pady = 50, 50
        # render everything wrt. the center of the screen
        cx, cy = (screen_width - 2*padx)/2, (screen_height - 2*pady)/2

        # set target rendering diameter relative to environment scale
        ball_dia = 0.025
        # set wall rendering thickness relative to environment scale
        wall_thickness = 0.05
        # set the world dimensions
        world_width, world_height = 1., 4.
        if self.arena:
            world_width, world_height = 1., 1.
        # calculate scaling factor
        scale = (screen_height - 2*pady) / world_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            if self.arena:
                # render walls
                left_wall_coords = [
                    [0., 0.], [-wall_thickness, 0.],
                    [-wall_thickness, 1.], [0., 1.], [0., 0.]
                ]
                top_wall_coords = [
                    [-wall_thickness, 1.], [-wall_thickness, 1.+wall_thickness],
                    [1.+wall_thickness, 1.+wall_thickness],
                    [1.+wall_thickness, 1.], [-wall_thickness, 1.]
                ]
                right_wall_coords = [
                    [1., 0.], [1.+wall_thickness, 0.],
                    [1. + wall_thickness, 1.], [1., 1.], [1., 0.]
                ]
                bottom_wall_coords = [
                    [-wall_thickness, 0.], [-wall_thickness, -wall_thickness],
                    [1.+wall_thickness, -wall_thickness], [1.+wall_thickness, 0.],
                    [-wall_thickness, 0.]
                ]

                def _to_screen_coords(p): return (
                    padx + cx + (p[0]-0.5)*scale, pady + cy + (p[1]-0.5)*scale)
                left_wall_coords = [_to_screen_coords(
                    p) for p in left_wall_coords]
                top_wall_coords = [_to_screen_coords(
                    p) for p in top_wall_coords]
                right_wall_coords = [_to_screen_coords(
                    p) for p in right_wall_coords]
                bottom_wall_coords = [_to_screen_coords(
                    p) for p in bottom_wall_coords]

                left_wall = rendering.make_polygon(left_wall_coords)
                top_wall = rendering.make_polygon(top_wall_coords)
                right_wall = rendering.make_polygon(right_wall_coords)
                bottom_wall = rendering.make_polygon(bottom_wall_coords)
                left_wall.set_color(0.5, 0.5, 0.9)
                top_wall.set_color(0.5, 0.5, 0.9)
                right_wall.set_color(0.5, 0.5, 0.9)
                bottom_wall.set_color(0.5, 0.5, 0.9)
                self.viewer.add_geom(left_wall)
                self.viewer.add_geom(top_wall)
                self.viewer.add_geom(right_wall)
                self.viewer.add_geom(bottom_wall)
            else:
                # render walls
                left_wall_coords = [
                    (padx + cx + (0.-0.5)*scale, 0.),
                    (padx + cx + (-0.1-0.5)*scale, 0.),
                    (padx + cx + (-0.1-0.5)*scale, screen_height),
                    (padx + cx + (0.-0.5)*scale, screen_height),
                    (padx + cx + (0.-0.5)*scale, 0.),
                ]
                right_wall_coords = [
                    (padx + cx + (1.-0.5)*scale, 0.),
                    (padx + cx + (1.1-0.5)*scale, 0.),
                    (padx + cx + (1.1-0.5)*scale, screen_height),
                    (padx + cx + (1.-0.5)*scale, screen_height),
                    (padx + cx + (1.-0.5)*scale, 0.),
                ]
                left_wall = rendering.make_polygon(left_wall_coords)
                right_wall = rendering.make_polygon(right_wall_coords)
                left_wall.set_color(0.5, 0.5, 0.9)
                right_wall.set_color(0.5, 0.5, 0.9)
                self.viewer.add_geom(left_wall)
                self.viewer.add_geom(right_wall)

            # spaceship coordinates relative to environment dimensions
            spaceship_coords = [
                (-0.02, -0.04), (0, 0.04), (0.02, -0.04), (-0.02, -0.04)
            ]
            # scale spaceship coordinates
            spaceship_coords = [(p[0]*scale, p[1]*scale)
                                for p in spaceship_coords]
            # render spaceship
            spaceship = rendering.make_polygon(spaceship_coords)
            spaceship.set_color(0, 0, 0)
            self.spaceship_trans = rendering.Transform()
            spaceship.add_attr(self.spaceship_trans)
            self.viewer.add_geom(spaceship)

            # render target
            target = rendering.make_circle(ball_dia*scale)
            target.set_color(0.8, 0., 0.)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            self.viewer.add_geom(target)

        self.target_trans.set_translation(
            padx + cx + (self.target_pos[0]-0.5)*scale, pady + cy + (self.target_pos[1]-0.5)*scale)

        angle = np.arctan2(
            self.spaceship_velocity[1], self.spaceship_velocity[0])
        self.spaceship_trans.set_rotation(angle - math.pi/2)

        self.spaceship_trans.set_translation(
            padx + cx + (self.spaceship_pos[0]-0.5)*scale, pady + cy + (self.spaceship_pos[1]-0.5)*scale)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
