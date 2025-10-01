import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ---- Toy Market Env ----
class MarketEnv:
    def __init__(self, horizon=100, sigma=0.01, cost=0.001):
        self.horizon = horizon
        self.sigma = sigma
        self.cost = cost
        self.reset()

    def reset(self):
        self.t = 0
        self.price = 100.0
        self.position = 0
        self.cash = 1000.0
        self.history = [0.0] * 10  # last 10 returns
        return self._obs()

    def step(self, action):
        # action: 0=hold, 1=buy, 2=sell
        if action == 0:  # buy
            self.position += 1
            self.cash -= self.price * (1 + self.cost)
        elif action == 1 and self.position > 0:  # sell
            self.position -= 1
            self.cash += self.price * (1 - self.cost)

        # evolve price
        ret = np.random.normal(0, self.sigma)
        self.price *= np.exp(ret)
        self.history = self.history[1:] + [ret]

        self.t += 1
        done = (self.t >= self.horizon)
        value = self.cash + self.position * self.price
        reward = value - (self.cash + self.position * self.price)  # delta, trivial here
        return self._obs(), reward, done, {}

    def _obs(self):
        return np.array(self.history + [self.price, self.position, self.cash], dtype=np.float32)

# ---- Policy + Value Net ----
class Net(nn.Module):
    def __init__(self, state_dim, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)

# ---- MCTS Node ----
class Node:
    def __init__(self, state, prior):
        self.state = state
        self.prior = prior
        self.value_sum = 0
        self.visit_count = 0
        self.children = {}

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

# ---- MCTS (simplified) ----
def mcts(env, net, state, n_sim=20):
    root = Node(state, 1.0)
    policy_logits, value = net(torch.tensor(state).unsqueeze(0))
    priors = torch.softmax(policy_logits, dim=1).detach().numpy()[0]

    # Expand root
    for a, p in enumerate(priors):
        root.children[a] = Node(None, p)

    # Run simulations
    for _ in range(n_sim):
        # Select action with max UCB
        best_a, best_child = max(root.children.items(),
            key=lambda kv: kv[1].value() + kv[1].prior / (1 + kv[1].visit_count))

        # Rollout 1 step in a cloned env
        clone = clone_env(env)
        s, r, done, _ = clone.step(best_a)
        with torch.no_grad():
            pol, val = net(torch.tensor(s).unsqueeze(0))
        v = val.item()
        # Backprop value
        best_child.value_sum += v
        best_child.visit_count += 1

    # Return improved policy = visit counts
    visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
    return visits / (np.sum(visits) + 1e-8)

def clone_env(env):
    clone = MarketEnv(env.horizon, env.sigma, env.cost)
    clone.t = env.t
    clone.price = env.price
    clone.position = env.position
    clone.cash = env.cash
    clone.history = list(env.history)
    return clone

# ---- Self-play ----
def self_play(env, net, n_games=5):
    data = []
    for _ in range(n_games):
        s = env.reset()
        trajectory = []
        done = False
        while not done:
            pi = mcts(env, net, s)
            a = np.random.choice(len(pi), p=pi)
            trajectory.append((s, pi))
            s, r, done, _ = env.step(a)
        final_value = env.cash + env.position * env.price
        for (state, pi) in trajectory:
            data.append((state, pi, final_value))
    return data

def live_run(env, net):
    s = env.reset()
    done = False
    values = []
    prices = []
    actions = []
    while not done:
        pi = mcts(env, net, s)
        a = np.argmax(pi)  # Take the most probable action
        actions.append(a)
        s, r, done, _ = env.step(a)
        values.append(env.cash + env.position * env.price)
        prices.append(env.price)
    print("Live run finished.")
    print("Final portfolio value:", values[-1])
    print("Actions taken:", actions)
    # Optional: plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        #plt.plot(prices, label='Price')
        plt.plot(values, label='Portfolio Value')
        plt.legend()
        plt.title('Simulated Live Run')
        plt.show()
    except ImportError:
        pass

# ---- Training ----
def train():
    env = MarketEnv()
    net = Net(len(env.reset()))
    opt = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(10):
        data = self_play(env, net)
        random.shuffle(data)
        for s, pi, z in data:
            s = torch.tensor(s).unsqueeze(0)
            pi = torch.tensor(pi).unsqueeze(0)
            z = torch.tensor([[z]], dtype=torch.float32)
            pol, val = net(s)
            loss = -(pi * torch.log_softmax(pol, dim=1)).sum() + (val - z).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} done.")
    live_run(env, net)
if __name__ == "__main__":
    train()
