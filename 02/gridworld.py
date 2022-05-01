from __future__ import annotations
from enum import IntEnum
import random
import numpy
import typing


class Gridworld:
    class Action(IntEnum):
        NORTH = 0
        EAST = 1
        SOUTH = 2
        WEST = 3

    def __init__(self, shape: typing.Tuple[int, int], init_pos: typing.Tuple[int, int], term_pos: typing.Tuple[int, int], term_reward: float = 1.0, block_chance: float = 0.1, action_fail_chance: float = 0.1, max_negative_reward: float = 1.0, seed: int = None) -> None:
        for val in shape:
            if val <= 0:
                raise ValueError

        if term_reward <= 0:
            print("The terminal reward must be strictly positive")
            raise ValueError

        self.rewards = numpy.zeros(shape, dtype=float)
        self.blocks = numpy.zeros(shape, dtype=bool)

        self.init_pos = init_pos
        self.term_pos = term_pos

        self._state = self.init_pos

        try:
            self.rewards[init_pos]
        except IndexError:
            print("Initial position out of bounds!")
            raise ValueError
        try:
            self.rewards[term_pos]
        except IndexError:
            print("Terminal position out of bounds!")
            raise ValueError

        self.term_reward = term_reward
        self.block_chance = block_chance
        self.action_fail_chance = action_fail_chance
        self.max_negative_reward = max_negative_reward
        
        self.seed = seed

        self._random_setup()

    def _random_setup(self):
        random.seed(self.seed)
        valid = False
        while not valid:
            self.blocks = numpy.zeros(self.blocks.shape, dtype=bool)
            for x in range(self.blocks.shape[0]):
                for y in range(self.blocks.shape[1]):
                    if not ((x, y) == self.init_pos or (x, y) == self.term_pos):
                        if random.random() < self.block_chance:
                            self.blocks[x, y] = True
                        self.rewards[x, y] = -1 * \
                            random.uniform(0, self.max_negative_reward)
            valid = self._check_validity()

        self.rewards[self.term_pos] = self.term_reward

        random.seed(None)

    def _check_validity(self):
        visited = set()
        queued = set()
        queued.add(self.term_pos)
        while len(queued) != 0:
            cur_pos = queued.pop()
            neighbors = self._get_neighbors(cur_pos)
            for neighbor in neighbors:
                if neighbor == self.init_pos:
                    return True
                if not self.blocks[neighbor] and neighbor not in visited:
                    queued.add(neighbor)
            visited.add(cur_pos)
        return False

    def _get_neighbors(self, pos: typing.Tuple[int, int]):
        neighbors = set()
        pos_neighbors = [(pos[0], pos[1] + 1), (pos[0], pos[1] - 1),
                         (pos[0] + 1, pos[1]), (pos[0] - 1, pos[1])]
        for neighbor in pos_neighbors:
            if neighbor[0] < self.blocks.shape[0] and neighbor[0] >= 0 and neighbor[1] < self.blocks.shape[1] and neighbor[1] >= 0:
                neighbors.add(neighbor)
        return neighbors

    def reset(self) -> typing.Tuple[int, int]:
        self._state = self.init_pos
        return self._state

    def step(self, action: Gridworld.Action) -> typing.Tuple[typing.Tuple[int, int], float, bool]:
        new_state = self._state_transition(action)
        term_state = new_state == self.term_pos
        reward = self.rewards[new_state[0], new_state[1]]

        self._state = new_state

        return new_state, reward, term_state

    def _state_transition(self, action: Gridworld.Action) -> typing.Tuple(int, int):
        if(random.random() > self.action_fail_chance):
            next_pos = [self._state[0], self._state[1]]
            next_pos[(action+1)%2] += int((action + 1) % 4 / 2)*2 - 1 #calculate position after action
            try:
                if(next_pos[0] < 0 or next_pos[1] < 0): raise IndexError
                if(not self.blocks[next_pos[0], next_pos[1]]):
                    return tuple(next_pos)
            except IndexError:
                pass
        valid = False
        while True:
            action = Gridworld.Action(random.randint(0, 3))
            next_pos = [self._state[0], self._state[1]]
            next_pos[(action+1)%2] += int((action + 1) % 4 / 2)*2 - 1 #calculate position after action
            try:
                if(next_pos[0] < 0 or next_pos[1] < 0): raise IndexError
                if(not self.blocks[next_pos[0], next_pos[1]]):
                    return tuple(next_pos)
            except IndexError:
                pass

    def visualize(self):
        print(self)

    def __repr__(self) -> str:
        retStr = ""
        for y in range(self.blocks.shape[0]):
            for x in range(self.blocks.shape[1]):
                if(self.blocks[x, y]):
                    retStr += " " + "X" * 5
                else:
                    red = int(min(0, self.rewards[x, y]/self.term_reward)*-255)
                    green = int(
                        max(0, self.rewards[x, y]/self.max_negative_reward) * 255)
                    yellow = int(255-max(red, green))
                    if(self._state[0] == x and self._state[1] == y):
                        retStr += colored(50, 50, 200, "{:>6s}".format("{:1.2f}".format(self.rewards[x, y])))
                    else:
                        retStr += colored(red+yellow, green+yellow, 0, "{:>6s}".format("{:1.2f}".format(self.rewards[x, y])))
            retStr += "\n"
        return retStr




def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m"


if __name__ == "__main__":
    grid = Gridworld((5, 5), (0, 0), (4, 4), 1, block_chance=0.6)
    print(grid)
    while(True):
        recv = input("wasd?")
        if(recv == "w"):
            action = Gridworld.Action.NORTH
        elif(recv == "a"):
            action = Gridworld.Action.WEST
        elif(recv == "s"):
            action = Gridworld.Action.SOUTH
        elif(recv == "d"):
            action = Gridworld.Action.EAST
        else:
            action = None
        grid.step(action)
        print(grid)
