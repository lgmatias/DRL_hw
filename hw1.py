from abc import             ABC, abstractmethod
from collections import     defaultdict
import gymnasium as         gym
import numpy as             np
import matplotlib.pyplot as plt
import torch as             t
from typing import          Literal

#
# runs all agents and builds comparison graph
#
def main():
    Agent(TD0)
    Agent(SARSA)
    Agent(QLearning)
    Agent(DQN)

    plt.legend()
    plt.xlabel("Episodes Total")
    plt.ylabel("Undiscounted Reward (Rolling Avg Across 25 Episodes)")
    plt.show()

class Algorithm(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.epsilon = 1.0
        self.epsilon_floor = 0.1
        self.epsilon_decay_rate = 0.96


    @abstractmethod
    def policy(self, observation: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, observation_old: np.ndarray, observation_new: np.ndarray, action_old:int, action_new:int, reward: int, terminated: bool) -> None:
        pass

    def discretize(self, observation: np.ndarray) -> np.ndarray: #returns state
        cart_posi_bin = np.linspace(-2.4,    2.4,    30)
        cart_velo_bin = np.linspace(-3,      3,      30)
        pole_angl_bin = np.linspace(-0.209,  0.209,  30)
        pole_angv_bin = np.linspace(-10,     10,     30)

        idx_posi = np.maximum(np.digitize(observation[0], cart_posi_bin)-1, 0)
        idx_velo = np.maximum(np.digitize(observation[1], cart_velo_bin)-1, 0)
        idx_angl = np.maximum(np.digitize(observation[2], pole_angl_bin)-1, 0)
        idx_angv = np.maximum(np.digitize(observation[3], pole_angv_bin)-1, 0)

        return tuple([float(idx_posi), float(idx_velo), float(idx_angl), float(idx_angv)])


class TD0(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "TD(0)"
        self.alpha = 1
        self.gamma = 1 #discount rate
        self.V = defaultdict(float) #V(s) map

    def policy(self, state: np.ndarray) -> Literal[0,1]:

        #take random action with epsilon probability
        if np.random.random() < self.epsilon: return np.random.choice([0,1])

        #adhere to greedy policy with 1 - epsilon probability
        return 0 if state[2] < 15 else 1

    def update(self, observation_old: np.ndarray, observation_new: np.ndarray, action_old: int, action_new: int, reward: int, terminated: bool) -> None:

        #update V(s)
        self.V[observation_old] += self.alpha * (reward + self.gamma*self.V[observation_new] - self.V[observation_old])

class SARSA(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "SARSA"
        self.alpha = 0.5
        self.gamma = 1
        self.epsilon_decay_rate = 0.96
        self.Q = defaultdict(lambda: np.zeros(2, float))

    def policy(self, state:np.ndarray) -> Literal[0,1]:
        
        #take random action with epsilon probability
        if np.random.random() < self.epsilon: return np.random.choice([0,1])

        #adhere to greedy policy with 1 - epsilon probability
        return np.argmax(self.Q[state])

    def update(self, state_old: np.ndarray, state_new: np.ndarray, action_old: int, action_new: int, reward: int, terminated: bool) -> None:

        #update Q(s,a)
        if terminated: self.Q[state_new][action_new] = 0
        self.Q[state_old][action_old] += self.alpha * (reward + self.gamma*self.Q[state_new][action_new] - self.Q[state_old][action_old])

class QLearning(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Q Learning"
        self.alpha = 0.5
        self.gamma = 1
        self.Q = defaultdict(lambda: np.zeros(2, float))

    def policy(self, state:np.ndarray) -> Literal[0,1]:
        
        #take random action with epsilon probability
        if np.random.random() < self.epsilon: return np.random.choice([0,1])

        #adhere to greedy policy with 1 - epsilon probability
        return np.argmax(self.Q[state])
    
    def update(self, state_old: np.ndarray, state_new: np.ndarray, action_old: int, action_new: int, reward: int, terminated: bool) -> None:

        #update Q(s,a)
        if terminated: self.Q[state_new][action_new] = 0
        self.Q[state_old][action_old] += self.alpha * (reward + self.gamma*np.max(self.Q[state_new][:]) - self.Q[state_old][action_old])

class DQN(Algorithm):
    def __init__(self) -> None:
        super().__init__()
        self.name = "DQN"
        self.alpha = 1
        self.gamma = 0.99
        self.nn = NeuralNetwork()
        self.mem_size = 100_000
        self.mem_iter = 0
        self.batch_size = 64

        self.mem_state =        np.zeros((self.mem_size, 4), dtype=np.float32)
        self.mem_state_new =    np.zeros((self.mem_size, 4), dtype=np.float32)
        self.mem_action =       np.zeros( self.mem_size,     dtype=np.  int32)
        self.mem_reward =       np.zeros( self.mem_size,     dtype=np.float32)
        self.mem_terminal =     np.zeros( self.mem_size,     dtype=      bool)

    def policy(self, state:np.ndarray) -> Literal[0,1]:

        #take random action with epsilon probability
        if np.random.random() < self.epsilon: return np.random.choice([0,1])

        #adhere to greedy policy with 1 - epsilon probability
        state = t.tensor(state).to(self.nn.device)
        actions = self.nn.forward(state)
        return t.argmax(actions).item()

    def update(self, state: np.ndarray, state_new: np.ndarray, action: int, action_new: int, reward: int, terminated: bool) -> None:
        i = self.mem_iter % self.mem_size
        self.mem_state      [i] = state
        self.mem_state_new  [i] = state_new
        self.mem_reward     [i] = reward
        self.mem_action     [i] = action
        self.mem_terminal   [i] = terminated
        self.mem_iter += 1

        if self.mem_iter < 50: return #dont learn if too early

        self.nn.opt.zero_grad()
        mem_size_local = min(self.mem_iter, self.mem_size)
        batch = np.random.choice(mem_size_local, self.batch_size, replace=True)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        batch_state =        t.tensor(self.mem_state        [batch]).to(self.nn.device)
        batch_state_new =    t.tensor(self.mem_state_new    [batch]).to(self.nn.device)
        batch_reward =       t.tensor(self.mem_reward       [batch]).to(self.nn.device)
        batch_terminal =     t.tensor(self.mem_terminal     [batch]).to(self.nn.device)
        batch_action =                self.mem_action       [batch]

        q_eval = self.nn.forward(batch_state)[batch_index, batch_action]
        q_next = self.nn.forward(batch_state_new)
        q_next[batch_terminal] = 0.0
        q_target = batch_reward + self.gamma * t.max(q_next, dim=1)[0]

        loss = self.nn.loss(q_target, q_eval).to(self.nn.device)
        loss.backward()
        self.nn.opt.step()

class NeuralNetwork(t.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr = 0.003
        self.fc1 = t.nn.Linear(4,   256)
        self.fc2 = t.nn.Linear(256, 256)
        self.fc3 = t.nn.Linear(256,   2)
        self.opt = t.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = t.nn.MSELoss()
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: t.Tensor):
        x = t.nn.functional.relu(self.fc1(state))
        x = t.nn.functional.relu(self.fc2(x))
        return self.fc3(x) #actions

#
# runs cart pole v1 simulation on the inputted algorithm
#
class Agent():
    def __init__(self, algorithm: Algorithm) -> None:

        #inits
        env = gym.make('CartPole-v1', render_mode='human')
        a = algorithm()
        reward_record: np.ndarray = np.zeros((500,), dtype=int)

        #agent/env loop for 500 episodes
        for i in range(500):

            #decrease epsilon over first ~100 trials
            if i > 25 and a.epsilon > a.epsilon_floor: a.epsilon *= a.epsilon_decay_rate
            if a.epsilon < a.epsilon_floor: a.epsilon = a.epsilon_floor

            observation = a.discretize(env.reset()[0])
            action = a.policy(observation)
            terminated = False; truncated = False; reward_sum = 0

            while not truncated and not terminated:
                observation_old = observation
                observation, reward, terminated, truncated, info = env.step(action)
                observation = a.discretize(observation)
                action_old = action
                action = a.policy(observation)
                a.update(observation_old, observation, action_old, action, reward, terminated)
                reward_sum += reward
            reward_record[i] = reward_sum
        plt.plot(self.rolling_avg(reward_record, 25), label = a.name)

    def rolling_avg(self, array: np.ndarray, n: int) -> None:
        r = np.cumsum(array, dtype=float)
        r[n:] = r[n:] - r[:-n]
        return r[n-1:]/n

main()