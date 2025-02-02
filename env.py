import enum
from typing import Any, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gameState
from game import Game
import typing
import math
import matplotlib.pyplot as plt

import pygame
import time

pygame.init()


class Display:
    SCALE = 20
    ROWS = 31
    COLS = 28

    class Shape(enum.Enum):
        CIRCLE = 0
        SQUARE = 1

    def __init__(self):
        size = (Display.SCALE * Display.COLS, Display.SCALE * Display.ROWS)
        self.window = pygame.display.set_mode(size)
        self.canvas = pygame.surface.Surface(size)

    def drawItem(
        self,
        canvas: pygame.Surface,
        color: typing.Tuple[float],
        position: typing.Tuple[int],
        shape: Shape,
        scale=1,
    ):
        row, col = position
        x, y = col * Display.SCALE, row * Display.SCALE
        width = height = Display.SCALE * scale
        centerX, centerY = x + Display.SCALE / 2, y + Display.SCALE / 2
        if shape == Display.Shape.CIRCLE:
            pygame.draw.circle(
                canvas, color, (centerX, centerY), Display.SCALE / 2 * scale
            )
        if shape == Display.Shape.SQUARE:
            pygame.draw.rect(
                canvas,
                color,
                pygame.rect.Rect(
                    centerX - width / 2, centerY - height / 2, width, height
                ),
            )

    def drawItems(self, canvas: pygame.surface.Surface, state: gameState.GameState):
        for row in range(Display.ROWS):
            for col in range(Display.COLS):
                pos = (row, col)
                config = None

                if state.wallAt(*pos):
                    config = (0, 0, 150), 1, Display.Shape.SQUARE
                if state.pelletAt(*pos):
                    config = (100, 100, 100), 0.25, Display.Shape.CIRCLE
                if state.superPelletAt(*pos):
                    config = (255, 255, 255), 0.5, Display.Shape.CIRCLE
                if state.fruitAt(*pos):
                    config = (255, 0, 0), 0.5, Display.Shape.CIRCLE

                if config is None:
                    continue

                color, scale, shape = config
                self.drawItem(canvas, color, pos, shape, scale)

    def drawEntities(self, canvas: pygame.surface.Surface, state: gameState.GameState):
        ghostColors = ((255, 0, 0), (255, 100, 100), (100, 100, 255), (255, 100, 0))

        freightened = (0, 0, 255)

        for ghost in state.ghosts:
            color = None
            pos = ghost.location.row, ghost.location.col
            if ghost.isFrightened():
                color = freightened
            else:
                color = ghostColors[ghost.color]
            self.drawItem(canvas, color, pos, Display.Shape.CIRCLE, 1)

        pos = state.pacmanLoc.row, state.pacmanLoc.col
        color = (255, 255, 0)
        self.drawItem(canvas, color, pos, Display.Shape.CIRCLE, 1)

    def render(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.canvas.fill((0, 0, 0))

        self.drawItems(self.canvas, state)
        self.drawEntities(self.canvas, state)

        self.window.blit(self.canvas, (0, 0))
        pygame.display.update()


TIME_PER_TICK = 1.5


class MotionProfilePacman(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode):
        if render_mode == "human":
            self.display = Display()

        self.game = Game()

        self.action = self.game.Action

        self.last_score = 0

        self.observation_space = spaces.Dict(
            {
                "pacbot_position": spaces.Box(0, 30, shape=(2,), dtype=int),
                "pink_ghost_position": spaces.Box(0, 30, shape=(2,), dtype=int),
                "blue_ghost_position": spaces.Box(0, 30, shape=(2,), dtype=int),
                "orange_ghost_position": spaces.Box(0, 30, shape=(2,), dtype=int),
                "red_ghost_position": spaces.Box(0, 30, shape=(2,), dtype=int),
                "pink_ghost_frightened_step": spaces.Discrete(41),
                "blue_ghost_frightened_step": spaces.Discrete(41),
                "orange_ghost_frightened_step": spaces.Discrete(41),
                "red_ghost_frightened_step": spaces.Discrete(41),
                "cherry_on": spaces.Discrete(2),
            }
        )

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([25, 4]), dtype=np.uint8) #(dist,direction), direction is same as ENUM

        # Motion constants
        self.max_vel = 3  # 3 blocks per second, placeholder
        self.max_accel = 1  # 1 block per second per second, placeholder

        self.currTime = 0.0

        self.vec_motion_profile = np.vectorize(
            self.motion_profile, excluded={"self", "start", "end"}
        )

    def reset(self, *args, **kwargs) -> Tuple[Any, dict]:
        self.game.reset()
        return self._get_obs(), {}

    def motion_profile(self, start: int, end: int, pos: int) -> float:
        """Receives the start position, end position, and the desired position. Outputs the real time when the bot arrives at that position"""
        # See the Pacbot Potential RL Model Discussion google doc, simulation requirement section for more information about the equations used
        # Most of these are just newton's motion equations
        if pos > end or pos < start:
            raise RuntimeError("position out of range")

        length = end - start

        if length <= self.max_vel**2 / self.max_accel:
            v_cap = math.sqrt(length * self.max_accel)
            # triangular profile because the distance is too short
            if pos <= length / 2:
                return math.sqrt((2 * pos) / self.max_accel)
            half_t = math.sqrt(length / self.max_accel)
            # pos = (end+start)/2 + vt - 1/2at^2, -1/2at^2 + vt + ((end+start)/2 - pos) = 0, 1/2at^2 - vt + (pos - (end+start)/2) = 0
            remaining_t = (
                v_cap
                - math.sqrt(
                    v_cap**2 - 4 * (0.5) * self.max_accel * (pos - (end + start) / 2)
                )
                / self.max_accel
            )
            return half_t + remaining_t

        # trapezoidal motion profile
        if pos <= 0.5 * self.max_vel**2 / self.max_accel:
            return math.sqrt((2 * pos) / self.max_accel)
        if pos <= length - 0.5 * self.max_vel**2 / self.max_accel:
            init_t = self.max_vel / self.max_accel
            remaining_t = (pos - 0.5 * self.max_vel**2 / self.max_accel) / self.max_vel
            return init_t + remaining_t

        init_t = self.max_vel / self.max_accel
        const_vel_t = (length - self.max_vel**2 / self.max_accel) / self.max_vel
        remaining_t = (
            self.max_vel
            - math.sqrt(
                self.max_vel**2
                - 4
                * (0.5)
                * self.max_accel
                * (pos - (length - 0.5 * self.max_vel**2 / self.max_accel))
            )
            / self.max_accel
        )

        return init_t + const_vel_t + remaining_t

    def step(self, action: Tuple) -> Tuple[Any | float | bool | dict]:
        """action should be a target location in format (row, col)"""

        pacloc = self.game.state.pacmanLoc

        action_dir = self.action.NONE
        move_dist = 0

        if action[0] == pacloc.row:
            if action[1] > pacloc.col:
                action_dir = self.action.RIGHT
                move_dist = action[1] - pacloc.col
            else:
                action_dir = self.action.LEFT
                move_dist = pacloc.col - action[1]
        else:
            if action[0] > pacloc.row:
                action_dir = self.action.DOWN
                move_dist = action[0] - pacloc.row
            else:
                action_dir = self.action.UP
                move_dist = pacloc.row - action[0]

        pos_list = np.linspace(1, move_dist, move_dist)
        t_list = self.vec_motion_profile(
            0, move_dist, pos_list
        )  # use broadcasting to vectorize and speed things up
        t_list[1:] -= t_list[:-1]  # get time difference between each time stamp

        for t in t_list:
            last_tick = math.floor(self.currTime / TIME_PER_TICK)
            new_tick = math.floor((self.currTime + t) / TIME_PER_TICK)
            for i in range(round(new_tick - last_tick)):
                self.game.update()
            self.game.step([action_dir])
            self.currTime += t
            self.render()
            time.sleep(0.5)  # Only for debugging

        observation = self._get_obs()
        reward = self._get_reward()
        done = self.game.state.currLives <= 0
        return observation, reward, done, False, {}

    def _get_obs(self):
        # self.state.update(ctypes.cast(self.obs_func(), ctypes.POINTER(ctypes.c_byte * 159)).contents)
        state = self.game.state
        ghosts = state.ghosts
        return {
            "pacbot_position": np.array([state.pacmanLoc.row, state.pacmanLoc.col]),
            "red_ghost_position": np.array(
                [ghosts[0].location.row, ghosts[0].location.col]
            ),
            "pink_ghost_position": np.array(
                [ghosts[1].location.row, ghosts[1].location.col]
            ),
            "blue_ghost_position": np.array(
                [ghosts[2].location.row, ghosts[2].location.col]
            ),
            "orange_ghost_position": np.array(
                [ghosts[3].location.row, ghosts[3].location.col]
            ),
            "red_ghost_frightened_step": ghosts[0].frightSteps,
            "pink_ghost_frightened_step": ghosts[1].frightSteps,
            "blue_ghost_frightened_step": ghosts[2].frightSteps,
            "orange_ghost_frightened_step": ghosts[3].frightSteps,
            "cherry_on": state.fruitSteps > 0,
        }

    def _get_reward(self):
        new_score = self.game.state.currScore
        reward = new_score - self.last_score
        self.last_score = new_score
        return reward

    def render(self):
        self.display.render(self.game.state)

    def close(self):
        # Close the environment and clean up resources
        self.game.reset()


if __name__ == "__main__":
    pac = MotionProfilePacman("human")
    time_list = []
    for i in range(16):
        time_list.append(pac.motion_profile(0, 15, i))
    plt.scatter(time_list, [i for i in range(16)])
    plt.show()

