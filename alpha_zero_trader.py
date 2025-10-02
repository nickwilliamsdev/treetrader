import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd

from utils.synthetic_data_service import SyntheticOHLCVGenerator
from utils.feature_functions import apply_features, apply_candlestick_features

# ---- Market Env with Data ----
class MarketEnv:
    def __init__(self, data, cost=0.001):
        self.data = data.reset_index(drop=True)
        self.horizon = len(data)
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
            features = row.drop(['time']).values.astype(np.float32)
        else:
            features = self.data.iloc[-1].drop(['time']).values.astype(np.float32)
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
    def __init__(self, input_dim, hidden_dim=64, action_dim=2):
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

# ---- MCTS (simplified) ----
def mcts(env, net, state, n_sim=20):
    root = Node(state, 1.0)
    state_tensor = torch.tensor(state)
    if state_tensor.ndim == 2:
        state_tensor = state_tensor.unsqueeze(0)
    policy_logits, value = net(state_tensor)
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
            s_tensor = torch.tensor(s)
            if s_tensor.ndim == 2:
                s_tensor = s_tensor.unsqueeze(0)
            pol, val = net(s_tensor)
        v = val.item()
        # Backprop value
        best_child.value_sum += v
        best_child.visit_count += 1

    # Return improved policy = visit counts
    visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
    return visits / (np.sum(visits) + 1e-8)

def clone_env(env):
    clone = MarketEnv(env.data.copy(), cost=env.cost)
    clone.t = env.t
    clone.price = env.price
    clone.position = env.position
    clone.cash = env.cash
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

# ---- Training Setup ----
def train_with_features(is_lstm=True):
    # 1. Generate synthetic OHLCV data
    generator = SyntheticOHLCVGenerator(n_steps=300, mu=0.0, sigma=0.0001, dt=1, seed=42)
    df = generator.generate(start=100)
    df.columns = [col.lower() for col in df.columns]
    # 2. Apply feature engineering
    df = apply_candlestick_features(df)

    # 3. Set up environment and network
    env = LSTMMarketEnv(df)
    obs = env.reset()
    input_dim = obs.shape[1]
    net = LSTMNet(input_dim, action_dim=2)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    # 4. Training loop
    if not is_lstm:
        for epoch in range(5):
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
    else:
        for epoch in range(5):
            data = self_play(env, net)
            random.shuffle(data)
            for s, pi, z in data:
                s = torch.tensor(s).unsqueeze(0)  # shape: (1, window, state_dim)
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
    train_with_features()
