import random

import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig


class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 1000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        lstmn = []
        mn = 10000
        while self.OPEN and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = (u.i + d[0], u.j + d[1])
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:
                    h = abs(n[0] - self.goal[0]) + abs(
                        n[1] - self.goal[1])  # Manhattan distance as a heuristic function
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor
                    if mn > h:
                        lstmn.clear()
                    lstmn.append((u.i, u.j))
                    if not h:
                        return
        if lstmn:
            self.goal = random.choice(lstmn)
            lstmn.clear()
        self.max_steps *= 1.5

    def get_next_node(self):
        next_node = self.start  # if path not found, current start position is returned
        if self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            while self.CLOSED[
                next_node] != self.start:  # get node in the path with start node as a predecessor
                next_node = self.CLOSED[next_node]
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(
            np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add(
                (n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(
            np.nonzero(other_agents))  # get the coordinates of all agents that are seen
        for agent in agents:
            if abs((n[0] - agent[0]) + abs(n[1] - agent[1])) < 14:
                self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))
            # self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))  # save them with correct coordinates


class Model:
    def __init__(self):
        self.agents = None
        self.kx = [0, -1, 1, 0, 0]
        self.ky = [0, 0, 0, -1, 1]
        self.lengthobs = 0
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(
                            len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.lengthobs = len(obs)
            self.agents = [AStar() for _ in range(self.lengthobs)]
        actions = []
        vec = set()
        for k in range(self.lengthobs):
            if positions_xy[k] == targets_xy[k]:
                actions.append(0)
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1],
                                            (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()
            pos = positions_xy[k]
            actions.append(self.actions[(next_node[0] - pos[0], next_node[1] - pos[1])])
            # print(actions)
            if set([(pos[0] + self.kx[i], pos[1] + self.ky[i]) for i in range(1, 5)]) & vec:
                actions[k] = 0

            vec.add((pos[0] + self.kx[actions[k]], pos[1] + self.ky[actions[k]]))
            vec.add(positions_xy[k])
        return actions


def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=32,  # ???????????????????? ?????????????? ???? ??????????
                             size=64,  # ?????????????? ??????????
                             density=0.3,  # ?????????????????? ??????????????????????
                             seed=1,  # ?????? ?????????????????? ??????????????
                             max_episode_steps=256,  # ???????????????????????? ?????????? ??????????????
                             obs_radius=5,  # ???????????? ????????????
                             )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # ?????????????????? ??????????????????
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0
    while not all(done):
        # ???????????????????? AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        steps += 1
        # print(steps, np.sum(done))

    # ?????????????????? ???????????????? ?? ???????????? ????
    env.save_animation("render.svg", egocentric_idx=None)


if __name__ == '__main__':
    main()
