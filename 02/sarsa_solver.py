import gridworld
import numpy
import random
import typing
import datetime
from enum import IntEnum

class SARSA_solver:
    def __init__(self, game: gridworld.Gridworld, epsilon: float = 0.1, alpha: float = 0.5, gamma: float = 0.2, n: int = 1, init_val: float = 0.0 ) -> None:
        self.n = n
        self.game = game
        self._state = game.reset()

        self._q_val = dict()
        self._init_val = init_val

        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
    
    def _step(self):
        best_action = (None, self._init_val)
        if random.random() < self._epsilon:
            best_action = random.choice(list(self.game.Action))
        else:
            for action in self.game.Action:
                try:
                    cur_action = (action, self._q_val[self._state[0], self._state[1], int(action)])
                    #print(str(type(cur_action[1])) + " vs " + str(type(best_action[1])))
                    if cur_action[1] > best_action[1]:
                        best_action = cur_action
                except KeyError:
                    pass
            if best_action[1] == self._init_val:
                best_action = random.choice(list(self.game.Action))
            else:   
                best_action = best_action[0]
            #print(best_action)
        new_state, reward, is_term = self.game.step(best_action)
        new_index = self._state[0], self._state[1], int(best_action)
        self._state = new_state
        return is_term, new_index, reward
    
    def __repr__(self) -> str:
        retStr = ""
        for y in range(self.game.blocks.shape[0]):
            val_str = ""
            for x in range(self.game.blocks.shape[1]):
                if(self.game.blocks[x, y]):
                    retStr += "█" * 6
                    val_str += "░" * 6
                else:
                    best_action = (None, numpy.NINF)
                    for action in self.game.Action:
                        try:
                            cur_action = (action, self._q_val[x, y, int(action)] )
                            #print(str(type(cur_action[1])) + " vs " + str(type(best_action[1])))
                            if cur_action[1] > best_action[1]:
                                best_action = cur_action
                        except KeyError:
                            pass

                    val_str += "{:6s}".format("{:1.2f}".format(best_action[1]))
                    best_action = best_action[0]
                    if(y == self.game.term_pos[1] and x == self.game.term_pos[0]):
                        retStr += "@@@".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.NORTH):
                        retStr += "↑".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.SOUTH):
                        retStr += "↓".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.EAST):
                        retStr += "→".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.WEST):
                        retStr += "←".center(6, " ")
                    else:
                        retStr += "{:6s}".format(" ")
            retStr += "\n" + val_str + "\n"
        return retStr

    def solve(self):
        self._state = self.game.reset()
        rewards = list()
        q_indices = list()
        finished, q_index, reward = self._step()
        q_indices.append(q_index)
        rewards.append(reward)
        while not finished:
            if(len(q_indices) >= self.n):
                q_index = q_indices.pop(0)
                # Calc return value for iteration
                ret = 0
                for i in range(len(rewards)):
                    ret += self._gamma ** i * rewards[i]
                try:
                    ret += self._gamma ** self.n * self._q_val[q_index]
                except KeyError:
                    ret += self._gamma ** self.n * self._init_val
                # Update Q
                try:
                    self._q_val[q_index] *= 1 - self._alpha
                except KeyError:
                    self._q_val[q_index] = self._init_val
                    self._q_val[q_index] *= 1 - self._alpha
                self._q_val[q_index] += self._alpha * ret
                rewards.pop(0)
            finished, q_index, reward = self._step()
            q_indices.append(q_index)
            rewards.append(reward)
        for q_index in q_indices:
            ret = 0
            for i in range(len(rewards)):
                ret += self._gamma ** i * rewards[i]
            # Update Q
            try:
                self._q_val[q_index] *= 1 - self._alpha
            except KeyError:
                self._q_val[q_index] = self._init_val
                self._q_val[q_index] *= 1 - self._alpha
            self._q_val[q_index] += self._alpha * ret
            
            rewards.pop(0)
        return

if __name__ == "__main__":
    size = (10, 10)
    game = gridworld.Gridworld(size, (0, 0), (random.randint(0, size[0] - 1), random.randint(0, size[1] - 1)), 10, block_chance = 0.4, max_negative_reward = 0, action_fail_chance = 0)
    solver = SARSA_solver(game, n = 100, epsilon = 0.2, gamma = 0.8, alpha = 0.3, init_val= -0.2)
    rep = 1000
    for i in range(rep):
        solver.solve()
        print(f'{datetime.datetime.now().isoformat()}: {i+1}/{rep}...')

    #print(solver.game)
    print(solver)
