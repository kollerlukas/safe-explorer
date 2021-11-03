import gym
from gym.spaces import Box
import numpy as np

import math

from safe_explorer.core.config import Config
from safe_explorer.env.constraintenv import ConstraintEnv


class BallND(ConstraintEnv):
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
        self.reward_shaping_margin = config.reward_shaping_margin
        # action space
        self.action_space = Box(
            low=-math.inf, high=math.inf, shape=(self.n,), dtype=np.double)
        # state space: ['ball position','ball velocity','target position']
        self.observation_space = Box(
            low=0., high=1., shape=(3*self.n,), dtype=np.double)
        # rendering viewer
        self.viewer = None

        # Sets all the episode specific variables
        self.reset()

    def reset(self, random_agent_position=False):
        if random_agent_position:
            self.ball_pos = (1 - 2 * self.agent_slack) * \
                np.random.random(self.n) + self.agent_slack
        else:
            self.ball_pos = 0.5 * np.ones(self.n)
        self.ball_velocity = np.zeros(self.n)
        self._reset_target_pos()
        self.time = 0.
        self.target_respawn_time = 0.
        return np.concatenate([self.ball_pos, self.ball_velocity, self._get_noisy_target_pos()])

    def _is_ball_outside_boundary(self):
        return np.any(self.ball_pos < 0.) or np.any(self.ball_pos > 1.)

    def _is_ball_outside_shaping_boundary(self):
        return np.any(self.ball_pos < self.reward_shaping_margin) \
            or np.any(self.ball_pos > 1 - self.reward_shaping_margin)

    def _get_reward(self):
        if self.enable_reward_shaping and self._is_ball_outside_shaping_boundary():
            return -1
        else:
            return np.clip(1 - 10 * np.linalg.norm(self.ball_pos - self.target_pos)**2, 0, 1)

    def _reset_target_pos(self):
        self.target_pos = (1 - 2 * self.target_margin) * \
            np.random.random(self.n) + self.target_margin

    def _move_ball(self):
        # advance time
        self.time += self.time_step
        self.target_respawn_time += self.time_step
        # move ball
        self.ball_pos += self.time_step * self.ball_velocity
        self.ball_velocity *= 0.95  # dampening
        # Check if the target needs to be relocated
        if self.target_respawn_time > self.respawn_interval:
            self._reset_target_pos()
            self.target_respawn_time = 0.

    def _get_noisy_target_pos(self):
        return self.target_pos + np.random.normal(0, self.target_noise_std, self.n)

    def get_num_constraints(self):
        return 2*self.n

    def get_constraint_values(self):
        # min_constraints = -self.ball_pos
        # max_constraints = self.ball_pos
        # return np.concatenate([min_constraints, max_constraints]), np.concatenate((np.repeat(-self.agent_slack, self.n), np.repeat(1 - self.agent_slack, self.n)))

        # define all constraint bounds as 0: C_i = 0 for all i
        # set lower and upper constraint for each dimension:
        #     slack < ball position < 1 - slack

        # slack < ball position --> slack - ball position < 0
        min_constraints = self.agent_slack - self.ball_pos
        # ball position < 1 - slack --> ball position - (1 - slack) < 0
        max_constraints = self.ball_pos - (1 - self.agent_slack)

        return np.concatenate([min_constraints, max_constraints])

    def step(self, action):
        # set action
        self.ball_velocity = action
        # execute 4 times steps; move ball 4 time steps
        for _ in range(0, 4):
            self._move_ball()
        # get reward
        reward = self._get_reward()
        # construct new state
        state = np.concatenate(
            [self.ball_pos, self.ball_velocity, self._get_noisy_target_pos()])
        # check for constraint violation
        constraint_violation = self._is_ball_outside_boundary()
        # check if done: (i) constraint violation or (ii) reached max episode length
        done = constraint_violation or self.time > self.episode_length

        return state, reward, done, {'constraint_violation': constraint_violation}

    def render(self, mode="human"):
        # only implemented render ball-1D & -3D domain for debugging purposes
        if self.n != 1 and self.n != 3:
            return None

        # set screen dimensions
        screen_width, screen_height = 600, 400
        # set padding from edges of the screen
        screen_padding = 50
        # render everything wrt. the center of the screen
        cx, cy = (screen_width - 2*screen_padding)/2, \
            (screen_height - 2*screen_padding)/2

        # set ball rendering diameter relative to environment scale
        ball_dia = 0.025
        # set the world dimensions
        world_width, world_height = 1., 1.
        if self.n == 3:
            world_width, world_height = 2., 2.  # compensate rotation of cube
        # calculate scaling factor
        scale = (screen_width - 2*screen_padding) / world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            if self.n == 1:
                # render line
                cube = rendering.Line(
                    start=(screen_padding + cx + (0.-0.5)*scale,
                           screen_padding + cy),
                    end=(screen_padding + cx + (1.-0.5)*scale,
                         screen_padding + cy))
                cube.set_color(0, 0, 0)
                self.viewer.add_geom(cube)
            elif self.n == 3:
                cube_polyline3d = [
                    [0, 0, 0], [1, 0, 0], [1, 1, 0],
                    [0, 1, 0], [0, 0, 0], [0, 0, 1],
                    [1, 0, 1], [1, 0, 0], [1, 0, 1],
                    [1, 1, 1], [1, 1, 0], [1, 1, 1],
                    [0, 1, 1], [0, 1, 0], [0, 1, 1],
                    [0, 0, 1], [0, 0, 0]
                ]
                cube_polyline2d = [self._project_point_3dTo2d(
                    p) for p in cube_polyline3d]
                cube_polyline2d = [(screen_padding + cx + (p[0]-0.5)*scale,
                                    screen_padding + cy + (p[1]-0.5)*scale) for p in cube_polyline2d]
                # render cube
                cube = rendering.make_polyline(cube_polyline2d)
                cube.set_color(0, 0, 0)
                self.viewer.add_geom(cube)

            # render target
            target = rendering.make_circle(ball_dia*scale)
            target.set_color(0.8, 0., 0.)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            self.viewer.add_geom(target)
            # render ball
            ball = rendering.make_circle(ball_dia*scale)
            ball.set_color(0., 0.8, 0.)
            self.ball_trans = rendering.Transform()
            ball.add_attr(self.ball_trans)
            self.viewer.add_geom(ball)
        # set current positions via translations
        if self.n == 1:
            self.ball_trans.set_translation(
                screen_padding + cx + (self.ball_pos[0]-0.5)*scale, screen_padding + cy)
            self.target_trans.set_translation(
                screen_padding + cx + (self.target_pos[0]-0.5)*scale, screen_padding + cy)
        elif self.n == 3:
            ball_pos2d = self._project_point_3dTo2d(self.ball_pos)
            self.ball_trans.set_translation(
                screen_padding + cx + (ball_pos2d[0]-0.5)*scale, screen_padding + cy + (ball_pos2d[1]-0.5)*scale)
            target_pos2d = self._project_point_3dTo2d(self.target_pos)
            self.target_trans.set_translation(
                screen_padding + cx + (target_pos2d[0]-0.5)*scale, screen_padding + cy + (target_pos2d[1]-0.5)*scale)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _project_point_3dTo2d(self, p):
        # rotation angles
        angle_x, angle_y, angle_z = math.pi/7, math.pi/3, 0
        # translation to origin
        translation = 0.5*np.ones(self.n)
        # rotation matrices
        rotate_x = np.matrix([
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x), math.cos(angle_x)]])

        rotate_y = np.matrix([
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)]])

        rotate_z = np.matrix([
            [math.cos(angle_z), -math.sin(angle_z), 0],
            [math.sin(angle_z), math.cos(angle_z), 0],
            [0, 0, 1]])
        # projection matrix to 2d
        proj_matr = np.matrix([
            [1, 0, 0],
            [0, 1, 0]])
        # apply transforms: translation + rotation + reverse translation
        p = p - translation
        p = np.asarray(np.dot(rotate_y, p)).squeeze()
        p = np.asarray(np.dot(rotate_x, p)).squeeze()
        p = np.asarray(np.dot(rotate_z, p)).squeeze()
        p = p + translation
        p = np.asarray(np.dot(proj_matr, p)).squeeze()
        # return projected point
        return p

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
