import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- Example architectures ---
class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        x = self.fc(x)
        attn_out, _ = self.attn(x, x, x)
        return attn_out[:, -1, :]

class ConvBlock(nn.Module):
    def __init__(self, input_dim, out_channels=32, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, out_channels, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)

# --- MuZero Core Networks ---
class RepresentationNet(nn.Module):
    def __init__(self, input_dim, arch='lstm'):
        super().__init__()
        if arch == 'lstm':
            self.block = LSTMBlock(input_dim)
        elif arch == 'attention':
            self.block = AttentionBlock(input_dim)
        elif arch == 'conv':
            self.block = ConvBlock(input_dim)
        else:
            raise ValueError("Unknown architecture")
    def forward(self, x):
        return self.block(x)

class DynamicsNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    def forward(self, state, action):
        # action: one-hot
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

class PredictionNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = nn.Linear(state_dim, action_dim)
        self.value = nn.Linear(state_dim, 1)
    def forward(self, state):
        return self.policy(state), self.value(state)

# --- Simple Trading Environment Example ---
class DummyTradingEnv:
    def __init__(self, input_dim, horizon=50):
        self.input_dim = input_dim
        self.horizon = horizon
        self.reset()
    def reset(self):
        self.t = 0
        self.price = 100.0
        self.position = 0
        self.cash = 1000.0
        self.data = np.random.randn(self.horizon, self.input_dim).astype(np.float32)
        return self._obs()
    def step(self, action):
        # action: 0=buy, 1=sell, 2=hold
        reward = 0.0
        if action == 0:  # buy
            self.position += 1
            self.cash -= self.price
        elif action == 1 and self.position > 0:  # sell
            self.position -= 1
            self.cash += self.price
        self.t += 1
        done = self.t >= self.horizon
        obs = self._obs()
        value = self.cash + self.position * self.price
        reward = value / 1000.0  # normalized
        return obs, reward, done, {}
    def _obs(self):
        if self.t < self.horizon:
            return self.data[self.t]
        else:
            return self.data[-1]

# --- MuZero Training Loop ---
def train_muzero(env, input_dim, action_dim, arch='lstm', epochs=10, unroll_steps=5):
    repr_net = RepresentationNet(input_dim, arch=arch)
    dyn_net = DynamicsNet(state_dim=64, action_dim=action_dim)
    pred_net = PredictionNet(state_dim=64, action_dim=action_dim)
    opt = optim.Adam(list(repr_net.parameters()) + list(dyn_net.parameters()) + list(pred_net.parameters()), lr=1e-3)

    for epoch in range(epochs):
        # 1. Collect trajectories (self-play)
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (batch, seq, input_dim)
            state = repr_net(obs_tensor)
            policy_logits, value = pred_net(state)
            policy = torch.softmax(policy_logits, dim=-1).detach().cpu().numpy()[0]
            action = np.random.choice(action_dim, p=policy)
            next_obs, reward, done, _ = env.step(action)
            trajectory.append((obs, action, reward))
            obs = next_obs

        # 2. Unroll MuZero targets
        for i in range(len(trajectory) - unroll_steps):
            obs_seq = []
            actions_seq = []
            rewards_seq = []
            for k in range(unroll_steps):
                obs_seq.append(trajectory[i + k][0])
                actions_seq.append(trajectory[i + k][1])
                rewards_seq.append(trajectory[i + k][2])
            # Initial state
            obs_tensor = torch.tensor(obs_seq[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            state = repr_net(obs_tensor)
            total_loss = 0.0
            for k in range(unroll_steps):
                # Prediction
                policy_logits, value_pred = pred_net(state)
                target_policy = torch.zeros(action_dim)
                target_policy[actions_seq[k]] = 1.0
                target_value = torch.tensor([rewards_seq[k]], dtype=torch.float32)
                # Losses
                policy_loss = nn.functional.cross_entropy(policy_logits, target_policy.unsqueeze(0))
                value_loss = nn.functional.mse_loss(value_pred, target_value.unsqueeze(0))
                total_loss += policy_loss + value_loss
                # Dynamics
                action_onehot = torch.zeros(1, action_dim)
                action_onehot[0, actions_seq[k]] = 1.0
                state = dyn_net(state, action_onehot)
            # 3. Optimize
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        print(f"Epoch {epoch} done.")
    print("Training complete.")

# --- Example usage ---
if __name__ == "__main__":
    input_dim = 10  # Number of features
    action_dim = 3  # e.g., buy/sell/hold
    env = DummyTradingEnv(input_dim)
    train_muzero(env, input_dim, action_dim, arch='lstm', epochs=10, unroll_steps=5)