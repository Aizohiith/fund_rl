import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.envs.registration import register
from collections import deque

class TSnake_Environment(gym.Env):
    """
    A simple Snake game environment for reinforcement learning.
    The snake moves around a grid, collecting food and avoiding collisions with walls and itself.
    Args:
        render_mode (str, optional): The mode for rendering the environment. Options are "human" and "rgb_array".
        max_episode_steps (int): Maximum number of steps per episode.
        size (tuple): The size of the grid (width, height).
        stacked_frames (int): Number of stacked frames to include in the observation.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 8,
        }

    def __init__(self, render_mode=None, max_episode_steps=1_000, size : tuple = (10, 10), stacked_frames : int = 1):
        super(TSnake_Environment, self).__init__()

        self.Food_Collection_Reward = 1
        self.Motivation_Reward = -0.01
        self.Die_Reward = -1
        self.render_mode = render_mode

        self.gs_Render_Mode = render_mode
        self.gi_Grid_Size = 64 * 10 // size[0]  # Size of each grid cell in pixels
        self.gi_Width = size[0]
        self.gi_Height = size[1]
        self.gi_Stacked_Observations = stacked_frames
        self.garr_Stacked_Observations = deque(maxlen=self.gi_Stacked_Observations)
        self.gi_Max_Steps = max_episode_steps
        self.gi_Window_Size = self.gi_Grid_Size * max(self.gi_Width, self.gi_Height)

        self.action_space = spaces.Discrete(4)

        self.spec = gym.envs.registration.EnvSpec(id="Snake-v0", entry_point=TSnake_Environment)

        self.observation_space = spaces.Box(low=0, high=1, shape=(16 * self.gi_Stacked_Observations,), dtype=np.int32 )

        self.reset()

    def reset(self, seed = None, options = None):
        """
        Resets the environment to an initial state and returns an initial observation.
        Args:
            seed (int, optional): The seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting the environment.
        """
        super().reset(seed=seed)
        self.larr_Snake = [(self.gi_Width // 2, self.gi_Height // 2)]
        self.Place_Food()
        self.gg_Direction = (0, 1)  # Initially moving right
        self.gb_Done = False
        self.gi_Step = 0

        for C1 in range(self.gi_Stacked_Observations):
            self.Observation()

        return self.Observation(), {}

    def step(self, pi_Action):
        """
        Executes one time step within the environment.
        Args:
            pi_Action (int): The action to be taken by the agent.
        Returns:
            observation (np.array): The observation of the environment after taking the action.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """
        self.gi_Step += 1

        if pi_Action == 0:  # Up
            self.gg_Direction = (-1, 0)
        elif pi_Action == 1:  # Down
            self.gg_Direction = (1, 0)
        elif pi_Action == 2:  # Left
            self.gg_Direction = (0, -1)
        elif pi_Action == 3:  # Right
            self.gg_Direction = (0, 1)

        ll_Head = self.larr_Snake[0]
        ll_New_Head = (ll_Head[0] + self.gg_Direction[0], ll_Head[1] + self.gg_Direction[1])

        # Check for collisions
        if (ll_New_Head in self.larr_Snake or ll_New_Head[0] < 0 or ll_New_Head[1] < 0 or ll_New_Head[0] >= self.gi_Height or ll_New_Head[1] >= self.gi_Width):
            self.gb_Done = True
            lf_Reward = self.Die_Reward
        else:
            self.larr_Snake.insert(0, ll_New_Head)

            if (len(self.larr_Snake) == self.gi_Width * self.gi_Height):
                self.gb_Done = True
                lf_Reward = self.Food_Collection_Reward
                self.gg_Food = None
            else:
                if ll_New_Head == self.gg_Food:
                    lf_Reward = self.Food_Collection_Reward
                    self.Place_Food()
                else:
                    self.larr_Snake.pop()
                    lf_Reward = self.Motivation_Reward

            
        return self.Observation(), lf_Reward, self.gb_Done, self.gi_Step > self.gi_Max_Steps, {}

    def render(self):
        """
        Renders the environment.
        """
        lb_Running = True
        if not (pygame.get_init()):
            pygame.init()

        if self.gs_Render_Mode == "human":
            if not hasattr(self, "gg_Screen"):
                self.gg_Screen = pygame.display.set_mode((self.gi_Window_Size, self.gi_Window_Size))
                self.clock = pygame.time.Clock()

            for ll_Event in pygame.event.get():
                if ll_Event.type == pygame.QUIT:
                    lb_Running = False

                if not lb_Running:  # Ensure pygame quits properly
                    env.close()
                    pygame.quit()
                    break

            self.gg_Screen.fill((0, 0, 0))

            # Draw snake
            for E1 in self.larr_Snake:
                pygame.draw.rect(
                    self.gg_Screen,
                    (0, 255, 0),
                    pygame.Rect(
                        E1[1] * self.gi_Grid_Size,
                        E1[0] * self.gi_Grid_Size,
                        self.gi_Grid_Size,
                        self.gi_Grid_Size,
                    ),
                )

            # Draw food
            pygame.draw.rect(
                self.gg_Screen,
                (255, 0, 0),
                pygame.Rect(
                    self.gg_Food[1] * self.gi_Grid_Size,
                    self.gg_Food[0] * self.gi_Grid_Size,
                    self.gi_Grid_Size,
                    self.gi_Grid_Size,
                ),
            )

            pygame.draw.rect(self.gg_Screen, (0, 155, 5), pygame.Rect(self.larr_Snake[0][1]*self.gi_Grid_Size, self.larr_Snake[0][0]*self.gi_Grid_Size, self.gi_Grid_Size, self.gi_Grid_Size))


            pygame.display.flip()
            self.clock.tick(8)

        elif self.gs_Render_Mode == "rgb_array":
            ll_Surface = pygame.Surface((self.gi_Window_Size, self.gi_Window_Size))
            ll_Surface.fill((0, 0, 0))

            for E1 in self.larr_Snake:
                pygame.draw.rect(ll_Surface, (0, 255, 0), pygame.Rect(E1[1]*self.gi_Grid_Size, E1[0]*self.gi_Grid_Size, self.gi_Grid_Size, self.gi_Grid_Size))
            if self.gg_Food:
                pygame.draw.rect(ll_Surface, (255, 0, 0), pygame.Rect(self.gg_Food[1]*self.gi_Grid_Size, self.gg_Food[0]*self.gi_Grid_Size, self.gi_Grid_Size, self.gi_Grid_Size))
            
            pygame.draw.rect(ll_Surface, (0, 155, 5), pygame.Rect(self.larr_Snake[0][1]*self.gi_Grid_Size, self.larr_Snake[0][0]*self.gi_Grid_Size, self.gi_Grid_Size, self.gi_Grid_Size))
            
            return pygame.surfarray.array3d(ll_Surface).swapaxes(0, 1)

    def close(self):
        """
        Closes the environment and cleans up resources.
        """
        if hasattr(self, "gg_Screen"):
            pygame.quit()

    def Observation(self):
        """
        Generates the current observation of the environment.
        Returns:
            observation (np.array): The current observation of the environment.
        """
        ll_Head = self.larr_Snake[0]
        ll_Dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        larr_Single_Observation = []

        for E1 in ll_Dir:
            li_Dx, li_Dy = E1
            li_X, li_Y = ll_Head
            li_Distance = -1
            ll_Obstacle_Type = [0, 0, 0]  # wall, snake, food

            while True:
                li_X += li_Dx
                li_Y += li_Dy
                li_Distance += 1

                if li_X < 0 or li_X >= self.gi_Height or li_Y < 0 or li_Y >= self.gi_Width:
                    ll_Obstacle_Type = [1, 0, 0]  # Wall
                    break

                if (li_X, li_Y) in self.larr_Snake:
                    ll_Obstacle_Type = [0, 1, 0]  # Snake
                    break

                if self.gg_Food == (li_X, li_Y):
                    ll_Obstacle_Type = [0, 0, 1]  # Food
                    break

            larr_Single_Observation.append(self.Flood_Fill(ll_Head[0] + li_Dx, ll_Head[1] + li_Dy) / (self.gi_Width * self.gi_Height))
            larr_Single_Observation.extend(ll_Obstacle_Type)
            

        self.garr_Stacked_Observations.append(larr_Single_Observation)

        larr_Observation = []
        for C1 in range(len(self.garr_Stacked_Observations)):
            larr_Observation.extend(self.garr_Stacked_Observations[C1])

        return np.array(larr_Observation, dtype=np.int32)

    def Place_Food(self):
        """
        Places food in a random location on the grid.
        """
        while True:
            ll_Food = (np.random.randint(0, self.gi_Height),np.random.randint(0, self.gi_Width))
            if ll_Food not in self.larr_Snake:
                self.gg_Food = ll_Food
                break

    def Flood_Fill(self, pi_X, pi_Y):
        """
        Performs a flood fill algorithm on the grid.
        """
        larr_Grid = np.zeros((self.gi_Height, self.gi_Width), dtype=np.int32)

        for E1 in self.larr_Snake:
            larr_Grid[E1[0], E1[1]] = 2

        larr_Grid[self.gg_Food[0], self.gg_Food[1]] = 3

        larr_Queue = deque()
        larr_Queue.append((pi_X, pi_Y))

        while len(larr_Queue) > 0:
            li_X, li_Y = larr_Queue.popleft()

            if li_X < 0 or li_X >= self.gi_Height or li_Y < 0 or li_Y >= self.gi_Width:
                continue

            if larr_Grid[li_X, li_Y] != 0 and larr_Grid[li_X, li_Y] != 3:
                continue

            larr_Grid[li_X, li_Y] = 1

            larr_Queue.append((li_X - 1, li_Y))
            larr_Queue.append((li_X + 1, li_Y))
            larr_Queue.append((li_X, li_Y - 1))
            larr_Queue.append((li_X, li_Y + 1))

        # Count the number of cells that are reachable
        return np.sum(larr_Grid == 1)

# Register the environment

register(
    id="Snake-v0",
    entry_point="fund_rl.environments.snake:TSnake_Environment"
)

if __name__ == "__main__":
    env = TSnake_Environment(render_mode="human")
    env.reset()
    pygame.init()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    li_Action = 0
                elif event.key == pygame.K_DOWN:
                    li_Action = 1
                elif event.key == pygame.K_LEFT:
                    li_Action = 2
                elif event.key == pygame.K_RIGHT:
                    li_Action = 3

                print(env.step(li_Action))
                if env.gb_Done:
                    env.reset()
    
            if not running:  # Ensure pygame quits properly
                env.close()
                pygame.quit()
                break

            env.render()

    env.close()