import time
from random import randint, shuffle

import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig

##########################################################################

class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class AStar:
    MAXSIZE = 64

    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.isbuild = False
        self.path = []

    def compute_shortest_path(self, start, goal, n=5):
        if self.isbuild:
            return
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        max_steps = self.max_steps # min(max(self.max_steps - 12 * n, 20), self.max_steps)
        while self.OPEN and steps < max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
                n = (u.i + d[0], u.j + d[1])
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:
                    h = abs(n[0] - self.goal[0]) + abs(
                        n[1] - self.goal[1])  # Manhattan distance as a heuristic function
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor
                    if not h:
                        return
    def check_new_pos(self, pos):
        return pos in self.obstacles or pos in self.other_agents
    def build(self):
        self.isbuild = False
        self.path.clear()
    def get_next_node(self):
        if not self.path:
            self.isbuild = False
        if self.isbuild:
            return self.path.pop()
        next_node = self.start  # if path not found, current start position is returned
        if self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            self.path.append(next_node)
            while self.CLOSED[next_node] != self.start:
                next_node = self.CLOSED[next_node]
                self.path.append(next_node)
            self.isbuild = True
            next_node = self.path.pop()
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(
            np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add(
                (n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(np.nonzero(other_agents))
        # get the coordinates of all agents that are seen
        for agent in agents:
            # mxs.add(abs((n[0] - agent[0]) + abs(n[1] - agent[1])))
            if abs((n[0] - agent[0]) + abs(n[1] - agent[1])) < randint(20, 150):
                self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))
            # save them with correct coordinates




class Model:
    MAXSIZE = 10

    def __init__(self):
        self.starttime = time.time()
        self.postmove = []

        self.positions = []
        self.kx = [0, -1, 1, 0, 0]
        self.ky = [0, 0, 0, -1, 1]
        self.kxy = [(self.kx[i], self.ky[i]) for i in range(len(self.kx))]
        self.failpair = {(1, 2), (2, 1), (3, 4), (4, 3), (0, 0)}
        self.agents = None
        self.key = False
        self.indexs = list()
        self.lst = list()
        self.lengthobs = 0
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(
                            len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id
        self.count = 0

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        #print(positions_xy)
        self.count += 1
        if self.count == 6:
            pass # breakpoint()
        if self.agents is None:
            self.lengthobs = len(dones)
            self.agents = [AStar() for _ in range(self.lengthobs)]
            self.postmove = [([0 for _ in range(self.lengthobs)]) for w in range(2)]
            self.positions = [([(-1, -1) for _ in range(self.lengthobs)]) for w in range(2)]
            self.lst = [0] * self.lengthobs

        actions = self.lst
        vec = set()
        lst = list(range(self.lengthobs))
        shuffle(lst)
        for k in lst:
            if dones[k]:  # positions_xy[k] == targets_xy[k]:
                self.postmove[self.key & 1][k] = 0
                actions[k] = 0
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1],
                                            (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k],
                                                 n=self.lengthobs)

            next_node = self.agents[k].get_next_node()
            actions[k] = self.actions.get((next_node[0] - positions_xy[k][0],
                                           next_node[1] - positions_xy[k][1]), 0)

            indx = actions[k]

            if set([(positions_xy[k][0] + self.kx[i], positions_xy[k][1] + self.ky[i]) for i in
                    range(1, 5)]) & vec:
                actions[k] = 0
                self.postmove[self.key & 1][k] = 0
            else:
                if positions_xy == [(20, 7)]:
                    pass # breakpoint()
                if not actions[k] or self.postmove[self.key & 1][k] == actions[k] and \
                        (actions[k], self.postmove[(self.key + 1) & 1][k]) not in self.failpair:
                    self.agents[k].build()

                    self.agents[k].update_obstacles(obs[k][0], obs[k][1],
                                                    (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
                    self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])

                    next_node = self.agents[k].get_next_node()
                    indx = actions[k] = self.actions.get((next_node[0] - positions_xy[k][0],
                                                   next_node[1] - positions_xy[k][1]), 0)

                shuffle(self.kxy)
                if not actions[k]:
                    for kx, ky in self.kxy:
                        i, j = positions_xy[k][0] + kx, positions_xy[k][1] + ky
                        if 0 <= i <= self.MAXSIZE and 0 <= j <= self.MAXSIZE:
                            if not (obs[k][0][i][j] or obs[k][1][i][j]):
                                if kx == ky:
                                    continue
                                indx = self.actions[(kx, ky)]
                                break
            actions[k] = self.postmove[self.key & 1][k] = indx

            vec.add(positions_xy[k])
        self.positions[self.key & 1] = positions_xy
        self.key = not self.key

        return actions[:self.lengthobs]


def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=100,  # количество агентов на карте
                             size=64,  # размеры карты
                             density=0.3,  # плотность препятствий
                             seed=1,  # сид генерации задания
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5,  # радиус обзора
                             )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0
    import time
    st = time.time()
    #print(st)
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        steps += 1
        # print(steps, np.sum(done))
    print((time.time() - st))
    # сохраняем анимацию и рисуем ее
    env.save_animation("render.svg", egocentric_idx=None)
    # print(max(mxs))


if __name__ == '__main__':
    main()
