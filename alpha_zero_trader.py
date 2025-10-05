import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd

from utils.synthetic_data_service import SyntheticOHLCVGenerator
from utils.feature_functions import apply_slope_features

# ---- Market Env with Data ----
class MarketEnv:
    def __init__(self, data, cost=0.001):
        self.data = data.reset_index(drop=True)
        self.horizon = 8
        self.cost = cost
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0
        self.cash = 1000.0
        self.price = float(self.data.loc[0, 'close'])
        return self._obs()

    def step(self, action):
        # action: 0=buy, 1=sell, 2=hold
        if action == 0:  # buy
            self.position += 1
            self.cash -= self.price * (1 + self.cost)
        elif action == 1 and self.position > 0:  # sell
            self.position -= 1
            self.cash += self.price * (1 - self.cost)
        # Advance to next timestep
        self.t += 1
        done = (self.t >= self.horizon)
        if not done:
            self.price = float(self.data.loc[self.t, 'close'])
        value = self.cash + self.position * self.price
        reward = value  # You can customize reward logic here
        return self._obs(), reward, done, {}

    def _obs(self):
        # Use all features except time as state
        if self.t < self.horizon:
            row = self.data.iloc[self.t]
            features = row.drop(['time'], errors='ignore').values.astype(np.float32)
        else:
            features = self.data.iloc[-1].drop(['time'], errors='ignore').values.astype(np.float32)
        return features

class LSTMMarketEnv:
    def __init__(self, data, cost=0.001, window=10):
        self.data = data.reset_index(drop=True)
        self.horizon = len(data)
        self.cost = cost
        self.window = window
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0
        self.cash = 1000.0
        self.price = float(self.data.loc[0, 'close'])
        return self._obs()

    def step(self, action):
        if action == 0:  # buy
            self.position += 1
            self.cash -= self.price * (1 + self.cost)
        elif action == 1 and self.position > 0:  # sell
            self.position -= 1
            self.cash += self.price * (1 - self.cost)
        self.t += 1
        done = (self.t >= self.horizon)
        if not done:
            self.price = float(self.data.loc[self.t, 'close'])
        value = self.cash + self.position * self.price
        reward = value
        return self._obs(), reward, done, {}

    def _obs(self):
        # Return last `window` rows as sequence
        start = max(0, self.t - self.window + 1)
        end = self.t + 1
        rows = self.data.iloc[start:end]
        features = rows.drop(['time'], axis=1, errors='ignore').values.astype(np.float32)
        # Pad if not enough history
        if len(features) < self.window:
            pad = np.zeros((self.window - len(features), features.shape[1]), dtype=np.float32)
            features = np.vstack([pad, features])
        return features

# ---- Policy + Value Net ----
class Net(nn.Module):
    def __init__(self, state_dim, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, action_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last hidden state
        return self.policy(out), self.value(out)

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

def simulate(env, net, state, depth):
    if depth == 0 or env.t >= env.horizon - 1:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        while state_tensor.ndim < 3:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            _, value = net(state_tensor)
        return value.item()
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        while state_tensor.ndim < 3:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = net(state_tensor)
        priors = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        a = np.random.choice(len(priors), p=priors)
        clone = clone_env(env)
        next_state, r, done, _ = clone.step(a)
        return simulate(clone, net, next_state, depth - 1)

# ---- MCTS (robust shape handling) ----
def mcts(env, net, state, n_sim=21, depth=5):
    root = Node(state, 1.0)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    while state_tensor.ndim < 3:
        state_tensor = state_tensor.unsqueeze(0)
    policy_logits, value = net(state_tensor)
    priors = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]

    # Expand root
    for a, p in enumerate(priors):
        root.children[a] = Node(None, p)

    # Run simulations
    for _ in range(n_sim):
        for a, child in root.children.items():
            clone = clone_env(env)
            next_state, r, done, _ = clone.step(a)
            v = simulate(clone, net, next_state, depth - 1)
            child.value_sum += v
            child.visit_count += 1

    visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
    return visits / (np.sum(visits) + 1e-8)

def clone_env(env):
    clone = LSTMMarketEnv(env.data.copy(), cost=env.cost, window=env.window)
    clone.t = env.t
    clone.price = env.price
    clone.position = env.position
    clone.cash = env.cash
    return clone

def self_play(env, net, n_games=8, max_steps=21):
    data = []
    for _ in range(n_games):
        # Start at a random index, ensure enough data for window and max_steps
        start_idx = random.randint(0, env.horizon - env.window - max_steps)
        env.t = start_idx
        env.position = 0
        env.cash = 1000.0
        env.price = float(env.data.loc[env.t, 'close'])
        s = env._obs()
        trajectory = []
        steps = 0
        done = False
        while not done and steps < max_steps and env.t < env.horizon - 1:
            pi = mcts(env, net, s)
            a = np.random.choice(len(pi), p=pi)
            trajectory.append((s, pi))
            s, r, done, _ = env.step(a)
            steps += 1
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
        a = np.argmax(pi)
        actions.append(a)
        s, r, done, _ = env.step(a)
        values.append(env.cash + env.position * env.price)
        prices.append(env.price)
    print("Live run finished.")
    print("Final portfolio value:", values[-1])
    print("Actions taken:", actions)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(values, label='Portfolio Value')
        plt.legend()
        plt.title('Simulated Live Run')
        plt.show()
    except ImportError:
        pass

# ---- Training Setup ----
def train_with_features(is_lstm=True):
    generator = SyntheticOHLCVGenerator(n_steps=1000, mu=0.005, sigma=0.05, dt=1, seed=72)
    df = generator.generate(start=100)
    df.columns = [col.lower() for col in df.columns]
    df = apply_slope_features(df, dropna=True)

    env = LSTMMarketEnv(df, window=10)
    obs = env.reset()
    input_dim = obs.shape[1]
    net = LSTMNet(input_dim, action_dim=2)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(5):
        data = self_play(env, net)
        random.shuffle(data)
        for s, pi, z in data:
            s_tensor = torch.tensor(s, dtype=torch.float32)
            while s_tensor.ndim < 3:
                s_tensor = s_tensor.unsqueeze(0)
            pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0)
            z = torch.tensor([[z]], dtype=torch.float32)
            pol, val = net(s_tensor)
            loss = -(pi * torch.log_softmax(pol, dim=1)).sum() + (val - z).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} done.")
    live_run(env, net)

if __name__ == "__main__":
    train_with_features()