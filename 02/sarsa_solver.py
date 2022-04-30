import gridworld
import numpy
import random
import typing
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

        pass
    
    def _step(self, old_index, old_reward):
        best_action = (None, self._init_val)
        if random.random()< self._epsilon:
            best_action = random.choice(list(self.game.Action))
        else:
            for action in self.game.Action:
                try:
                    cur_action = (action, self._q_val[self._state[0], self._state[1], int(action)] )
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
        try:
            change = float(old_reward) + self._gamma * self._q_val[new_index] - self._q_val[old_index]
        except KeyError:
            self._q_val[new_index] = self._init_val
            change = float(old_reward) + self._gamma * self._q_val[new_index] - self._q_val[old_index]
        self._q_val[old_index] += self._alpha * change 
        self._state = new_state
        return is_term, new_index, reward
    
    def __repr__(self) -> str:
        retStr = ""
        for y in range(self.game.blocks.shape[0]):
            val_str = ""
            for x in range(self.game.blocks.shape[1]):
                if(self.game.blocks[x, y]):
                    retStr += "X" * 6
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
                    if(best_action == gridworld.Gridworld.Action.NORTH):
                        retStr += "^".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.SOUTH):
                        retStr += "_".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.EAST):
                        retStr += ">".center(6, " ")
                    elif(best_action == gridworld.Gridworld.Action.WEST):
                        retStr += "<".center(6, " ")
                    else:
                        retStr +="{:6s}".format("")
            retStr += "\n"
            retStr += val_str + "\n"
        return retStr

    def solve(self):
        old_index = (0,0,0)
        self._q_val[old_index] = self._init_val
        steps = []
        for i in range(self.n):
            self._state = self.game.reset()
            steps.append(0)
            #print(self.game)
            finished = False
            old_index = (0,0,0)
            old_reward = 0.0
            while(not finished):
                finished, old_index, old_reward = self._step(old_index, old_reward)
                #print(self.game)
                steps[-1] += 1
                pass

            change = float(old_reward) + self._gamma * self._q_val[old_index] - self._q_val[old_index]
            self._q_val[old_index] += self._alpha * change 
            #print(steps[-1])
        #print(steps)
        


if __name__ == "__main__":
    game = gridworld.Gridworld((10, 10), (0, 0), (9, 9), 1, block_chance=0.1)
    solver = SARSA_solver(game, n = 1000, epsilon=0.5, gamma=0.8)
    solver.solve()
    print(solver)
