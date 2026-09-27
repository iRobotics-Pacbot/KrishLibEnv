import enum
from typing import Any, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gameState
from game import Game
import typing
import math

import walls

import pygame

pygame.init()


class Display:
    SCALE = 20
    ROWS = 31
    COLS = 28

    class Shape(enum.Enum):
        CIRCLE = 0
        SQUARE = 1

    def __init__(self, surface=None):
        size = (Display.SCALE * Display.COLS, Display.SCALE * Display.ROWS)
        if surface is None:
            self.window = pygame.display.set_mode(size)
        else:
            self.window = surface

    def drawItem(
        self,
        window: pygame.Surface,
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
                window, color, (centerX, centerY), Display.SCALE / 2 * scale
            )
        if shape == Display.Shape.SQUARE:
            pygame.draw.rect(
                window,
                color,
                pygame.rect.Rect(
                    centerX - width / 2, centerY - height / 2, width, height
                ),
            )

    def drawItems(self, window: pygame.surface.Surface, state: gameState.GameState):
        for row in range(Display.ROWS):
            for col in range(Display.COLS):
                pos = (row, col)
                config = None

                if state.wallAt(*pos):
                    config = (0, 0, 150), 1, Display.Shape.SQUARE
                if state.pelletAt(*pos):
                    config = (100, 100, 100), 0.5, Display.Shape.CIRCLE
                if state.superPelletAt(*pos):
                    config = (255, 255, 255), 0.75, Display.Shape.CIRCLE
                if state.fruitAt(*pos):
                    config = (255, 0, 0), 0.5, Display.Shape.CIRCLE

                if config is None:
                    continue

                color, scale, shape = config
                self.drawItem(window, color, pos, shape, scale)

    def drawEntities(self, window: pygame.surface.Surface, state: gameState.GameState):
        ghostColors = ((255, 0, 0), (255, 100, 100), (100, 100, 255), (255, 100, 0))

        freightened = (0, 0, 255)

        for ghost in state.ghosts:
            color = None
            pos = ghost.location.row, ghost.location.col
            if ghost.isFrightened():
                color = freightened
            else:
                color = ghostColors[ghost.color]
            self.drawItem(window, color, pos, Display.Shape.CIRCLE, 1)

        pos = state.pacmanLoc.row, state.pacmanLoc.col
        color = (255, 255, 0)
        self.drawItem(window, color, pos, Display.Shape.CIRCLE, 1)

    def render(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((0, 0, 0))

        self.drawItems(self.window, state)
        self.drawEntities(self.window, state)
        font = pygame.font.SysFont("Arial", 24)
        score_surface = font.render(f"Score: {state.currScore}", True, (255, 255, 255))
        self.window.blit(score_surface, (10, 10))
        pygame.display.update()


TIME_PER_TICK = 1.5
MOVEMENT_TICK = 10


class MotionProfilePacman(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode):
        self.screen = pygame.display.set_mode(
            (Display.SCALE * Display.COLS, Display.SCALE * Display.ROWS)
        )
        if render_mode == "human":
            self.display = Display(self.screen)

        self.render_mode = render_mode

        self.game = Game()

        self.action = self.game.Action

        self.last_score = 0

        self.observation_space = spaces.Dict(
            {
                "pacbot_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "pink_ghost_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "blue_ghost_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "orange_ghost_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "red_ghost_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "pink_ghost_frightened_step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "blue_ghost_frightened_step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "orange_ghost_frightened_step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "red_ghost_frightened_step": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                # "cherry_on": spaces.Discrete(2),
                "cherry_on": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "nearest_pellet": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "nearest_power_pellet": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "game_mode": spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32),
                "local_walls": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                # "board": spaces.Box(low=-1, high=5, shape=(31, 28), dtype=np.float32),
            }
        )

        # self.action_space = spaces.MultiDiscrete(
        #     [26, 5]
        # )  # (dist,direction), direction is same as ENUM
        # self.action_space = spaces.Discrete(130)
        self.action_space = spaces.Discrete(5)

        # Motion constants
        self.max_vel = 3  # 3 blocks per second, placeholder
        self.max_accel = 1  # 1 block per second per second, placeholder

        self.currTime = 0.0

        # self.vec_motion_profile = np.vectorize(
        #     self.motion_profile, excluded={"self", "start", "end"}
        # )

        self.last_lives = 3

    def reset(self, *args, **kwargs) -> Tuple[Any, dict]:
        self.game.reset()
        self.game.update()
        obs = self._get_obs()
        self.visited_positions = set()
        self.last_lives = 3
        self.last_score = 0

        nearest_pellet_pos = self.find_nearest_pellet(self.game.state.pacmanLoc)
        self.last_dist_to_pellet = abs(nearest_pellet_pos[0] - self.game.state.pacmanLoc.row) + \
                                   abs(nearest_pellet_pos[1] - self.game.state.pacmanLoc.col)
        return (obs, {})

    def motion_profile(self, start: int, end: int, pos: int) -> float:
        """Receives the start position, end position, and the desired position. Outputs the real time when the bot arrives at that position"""
        # See the Pacbot Potential RL Model Discussion google doc, simulation requirement section for more information about the equations used
        # Most of these are just newton's motion equations
        if pos > end or pos < start:
            raise RuntimeError("position out of range")

        length = end - start

        if length < 0:
            raise ValueError("Incorrect starting and end points")

        if length <= self.max_vel**2 / self.max_accel:
            v_cap = math.sqrt(length * self.max_accel)
            # triangular profile because the distance is too short
            if pos <= length / 2:
                return math.sqrt((2 * pos) / self.max_accel)
            half_t = math.sqrt(length / self.max_accel)
            # pos = (end+start)/2 + vt - 1/2at^2, -1/2at^2 + vt + ((end+start)/2 - pos) = 0, 1/2at^2 - vt + (pos - (end+start)/2) = 0
            temp = v_cap**2 - 4 * (0.5) * self.max_accel * (pos - (end + start) / 2)
            if temp < -1e-5:
                raise ValueError("Negative squareroot value")
            else:
                temp = abs(round(temp, 8))
            remaining_t = v_cap - math.sqrt(temp) / self.max_accel
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

    def max_dist_in_dir(self, dist: int, dir: int):
        if dir == self.action.NONE:
            return 0
        pacloc = self.game.state.pacmanLoc
        if pacloc.row == 32:
            return 0  # not checking for columns because when it shifts out of bounds automatically equals to 0
        for i in range(dist):
            match dir:
                case self.action.UP:
                    if walls.get(pacloc.row - i, pacloc.col):
                        return i - 1
                case self.action.DOWN:
                    if walls.get(pacloc.row + i, pacloc.col):
                        return i - 1
                case self.action.LEFT:
                    if walls.get(pacloc.row, pacloc.col - i):
                        return i - 1
                case self.action.RIGHT:
                    if walls.get(pacloc.row, pacloc.col + i):
                        return i - 1
        return 0

    def step(self, action: Tuple[int, int]) -> Tuple[Any, float, bool, bool, dict]:
        """
        old version:action should be a target location in format (row, col)
        new versoin:action shoudl be (dist,dir)
        """
        action_dir = [e for e in self.action][int(action)]
        
        state = self.game.state
        r, c = state.pacmanLoc.row, state.pacmanLoc.col
        hit_wall = False

        if action_dir == self.action.UP and state.wallAt(r - 1, c):
            hit_wall = True
        elif action_dir == self.action.DOWN and state.wallAt(r + 1, c):
            hit_wall = True
        elif action_dir == self.action.LEFT and state.wallAt(r, c - 1):
            hit_wall = True
        elif action_dir == self.action.RIGHT and state.wallAt(r, c + 1):
            hit_wall = True

        if hit_wall:
            action_dir = self.action.NONE
        
        for i in range(MOVEMENT_TICK):
            self.game.update()
        self.game.step([action_dir])
        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        reward = self._get_reward(hit_wall)
        done = self.game.state.currLives <= 0
        info = {}
        if done:
            info["terminal_observation"] = observation
            info["final_score"] = self.game.state.currScore
        # print(observation)
        return observation, reward, done, False, info

    def find_nearest_pellet(self, pacman_loc):
        state = self.game.state
        nearest_dist = float('inf')
        # Default to center of map if no pellets left (to avoid crash)
        nearest_pos = (15, 14) 

        # Loop through every row and column to find pellets
        for r in range(31):
            row_data = state.pelletArr[r]
            if row_data == 0:
                continue # Skip empty rows for speed
            
            for c in range(28):
                # Check if bit 'c' is 1 (meaning a pellet is there)
                if (row_data >> c) & 1:
                    # Manhattan distance is faster for the AI to understand in a grid
                    dist = abs(r - pacman_loc.row) + abs(c - pacman_loc.col)
                    
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_pos = (r, c)
        
        return nearest_pos

    def find_nearest_power_pellet(self, pacman_loc):
        state = self.game.state
        nearest_dist = float('inf')
        # Default to center of map if no pellets left (to avoid crash)
        nearest_pos = (15, 14) 

        power_pellet_coords = [(3, 1), (3, 26), (23, 1), (23, 26)]

        
        for coord in power_pellet_coords:
            r, c = coord
            row_data = state.pelletArr[r]
            if (row_data >> c) & 1:
                # Manhattan distance is faster for the AI to understand in a grid
                dist = abs(r - pacman_loc.row) + abs(c - pacman_loc.col)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pos = (r, c)
        
        return nearest_pos

    def _get_obs(self):
        # self.state.update(ctypes.cast(self.obs_func(), ctypes.POINTER(ctypes.c_byte * 159)).contents)
        state = self.game.state
        ghosts = state.ghosts
        nearest_pellet_pos = self.find_nearest_pellet(state.pacmanLoc)
        rel_row = (nearest_pellet_pos[0] - state.pacmanLoc.row) / 31.0
        rel_col = (nearest_pellet_pos[1] - state.pacmanLoc.col) / 28.0
        
        nearest_power_pellet_pos = self.find_nearest_power_pellet(state.pacmanLoc)
        power_rel_row = (nearest_power_pellet_pos[0] - state.pacmanLoc.row) / 31.0
        power_rel_col = (nearest_power_pellet_pos[1] - state.pacmanLoc.col) / 28.0
        
        # board = np.zeros((31, 28), dtype=np.float32)
        
        # for r in range(31):
        #     row_data = state.pelletArr[r]
        #     for c in range(28):
        #         if state.wallAt(r, c):
        #             board[r, c] = -1.0
        #         elif state.superPelletAt(r, c):
        #             board[r, c] = 2.0 
        #         elif (row_data >> c) & 1:
        #             board[r, c] = 1.0
        # if state.pacmanLoc.row < 31 and state.pacmanLoc.col < 28:
        #     board[state.pacmanLoc.row, state.pacmanLoc.col] = 5.0

        # for ghost in ghosts:
        #     gr, gc = ghost.location.row, ghost.location.col
        #     if gr < 31 and gc < 28:
        #         if ghost.isFrightened():
        #             board[gr, gc] = 3.0
        #         else:
        #             board[gr, gc] = -5.0

        r = state.pacmanLoc.row
        c = state.pacmanLoc.col
        wall_up = float(state.wallAt(r - 1, c))
        wall_down = float(state.wallAt(r + 1, c))
        wall_left = float(state.wallAt(r, c - 1) or (r == 14 and c == 0))
        wall_right = float(state.wallAt(r, c + 1) or (r == 14 and c == 27))


        return {
            "pacbot_position": np.array([state.pacmanLoc.row / 31.0, state.pacmanLoc.col / 28.0], dtype=np.float32),
            "red_ghost_position": np.array([ghosts[0].location.row / 31.0, ghosts[0].location.col / 28.0], dtype=np.float32),
            "pink_ghost_position": np.array([ghosts[1].location.row / 31.0, ghosts[1].location.col / 28.0], dtype=np.float32),
            "blue_ghost_position": np.array([ghosts[2].location.row / 31.0, ghosts[2].location.col / 28.0], dtype=np.float32),
            "orange_ghost_position": np.array([ghosts[3].location.row / 31.0, ghosts[3].location.col / 28.0], dtype=np.float32),
            "red_ghost_frightened_step": np.array([ghosts[0].frightSteps / 40.0], dtype=np.float32),
            "pink_ghost_frightened_step": np.array([ghosts[1].frightSteps / 40.0], dtype=np.float32),
            "blue_ghost_frightened_step": np.array([ghosts[2].frightSteps / 40.0], dtype=np.float32),
            "orange_ghost_frightened_step": np.array([ghosts[3].frightSteps / 40.0], dtype=np.float32),
            # "cherry_on": int(state.fruitSteps > 0),
            "cherry_on": np.array([float(state.fruitSteps > 0)], dtype=np.float32),
            "nearest_pellet": np.array([rel_row, rel_col], dtype=np.float32),
            "nearest_power_pellet": np.array([power_rel_row, power_rel_col], dtype=np.float32),
            "game_mode": np.array([float(state.gameMode)], dtype=np.float32),
            "local_walls": np.array([wall_up, wall_down, wall_left, wall_right], dtype=np.float32),
            # "board": board,
        }

    def _get_reward(self, hit_wall=False):
        state = self.game.state
        new_score = self.game.state.currScore

        actual_score_gain = float(new_score - self.last_score)
        reward = actual_score_gain
        self.last_score = new_score
        current_lives = self.game.state.currLives
        if current_lives < self.last_lives: # If lives decreased
            reward -= 200
        self.last_lives = current_lives
        
        if actual_score_gain >= 200:
            # reward += (actual_score_gain)
            reward += actual_score_gain * 2.0
        if actual_score_gain > 0:
            reward += 30
        reward -= 0.1
        if hit_wall:
            reward -= 5.0
        for ghost in self.game.state.ghosts:
            dist = abs(state.pacmanLoc.row - ghost.location.row) + abs(state.pacmanLoc.col - ghost.location.col)
            if not ghost.isFrightened():
                if dist < 2:
                    reward -= 10.0
            else:
                if dist < 5:
                    # reward += 10.0
                    reward += (1.0 / (dist + 1)) * 15.0
                    # reward += (1.0 / (dist_to_frightened_ghost + 1)) * 15.0
        nearest_pellet_pos = self.find_nearest_pellet(state.pacmanLoc)
        dist_to_pellet = abs(nearest_pellet_pos[0] - state.pacmanLoc.row) + \
                         abs(nearest_pellet_pos[1] - state.pacmanLoc.col)
        
        # Give a small reward for being close to food
        # reward += (1.0 / (dist_to_pellet + 1)) * 2.0
        if dist_to_pellet < self.last_dist_to_pellet:
            reward += 0.5
        elif dist_to_pellet > self.last_dist_to_pellet:
            reward -= 0.5
            
        self.last_dist_to_pellet = dist_to_pellet

        return reward

    def render(self):
        self.display.render(self.game.state)

    def _get_frame(self):
        # Get the pixel array from the screen
        return pygame.surfarray.array3d(self.screen)

    def close(self):
        # Close the environment and clean up resources
        self.reset()


if __name__ == "__main__":
    pass
