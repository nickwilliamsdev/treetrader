import os
import random
import math
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from api_wrappers.kraken_wrapper import KrakenWrapper
from utils.feature_functions import apply_slope_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Architectures
# -----------------------
class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # (B, hidden_dim)

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        x = self.fc(x)
        attn_out, _ = self.attn(x, x, x)
        return attn_out[:, -1, :]

class ConvBlock(nn.Module):
    def __init__(self, input_dim, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, out_channels, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)

# -----------------------
# MuZero nets: representation, dynamics (with reward), prediction
# -----------------------
class RepresentationNet(nn.Module):
    def __init__(self, input_dim, arch='lstm', hidden_dim=64):
        super().__init__()
        if arch == 'lstm':
            self.block = LSTMBlock(input_dim, hidden_dim)
        elif arch == 'attention':
            self.block = AttentionBlock(input_dim, hidden_dim)
        elif arch == 'conv':
            self.block = ConvBlock(input_dim, out_channels=hidden_dim)
        else:
            raise ValueError("Unknown arch")
    def forward(self, obs):  # obs: (B, seq_len, input_dim)
        return self.block(obs)  # (B, hidden_dim)

class DynamicsNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
    def forward(self, state, action_onehot):  # state: (B, state_dim)
        x = torch.cat([state, action_onehot], dim=-1)
        h = self.fc1(x)
        next_s = self.next_state(h)
        reward = self.reward_head(h)
        return next_s, reward

class PredictionNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = nn.Linear(state_dim, action_dim)
        self.value = nn.Linear(state_dim, 1)
    def forward(self, state):
        return self.policy(state), self.value(state)

# -----------------------
# Environment wrapper (lightweight LSTMMarketEnv)
# -----------------------
class LSTMMarketEnv:
    def __init__(self, df, window=21, cost=0.0, initial_cash=1000.0, max_position=10):
        # df: pandas DataFrame with features, must include 'close' column
        self.data = df.reset_index(drop=True)
        self.horizon = len(self.data)
        self.window = window
        self.cost = cost
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.reset()

    def reset(self, start_idx=0):
        self.t = int(start_idx)
        self.position = 0
        self.cash = float(self.initial_cash)
        self.price = float(self.data.loc[self.t, 'close'])
        self.max_portfolio_value = self.cash
        return self._obs()

    def _obs(self):
        start = max(0, self.t - self.window + 1)
        rows = self.data.iloc[start:self.t+1].drop(columns=['date'], errors='ignore').values.astype(np.float32)
        if len(rows) < self.window:
            pad = np.zeros((self.window - len(rows), rows.shape[1]), dtype=np.float32)
            rows = np.vstack([pad, rows])
        return rows  # shape: (window, feature_dim)

    def step(self, action):
        # action: 0=buy, 1=sell
        prev_value = self.cash + self.position * self.price
        # buy only if enough cash and within position limit
        if action == 0 and self.cash >= self.price * (1 + self.cost) and self.position < self.max_position:
            self.position += 1
            self.cash -= self.price * (1 + self.cost)
        # sell only if position > 0
        elif action == 1 and self.position > 0:
            self.position -= 1
            self.cash += self.price * (1 - self.cost)
        # advance
        self.t += 1
        done = (self.t >= self.horizon)
        if not done:
            self.price = float(self.data.loc[self.t, 'close'])
        value = self.cash + self.position * self.price
        # immediate P&L reward (relative) and penalize drawdown
        reward = (value - prev_value) / (prev_value + 1e-8)
        self.max_portfolio_value = max(self.max_portfolio_value, value)
        drawdown = (self.max_portfolio_value - value) / (self.max_portfolio_value + 1e-8)
        reward = reward - 0.5 * drawdown  # tune weight
        return self._obs(), float(reward), done, {}

# -----------------------
# MCTS Implementation (MuZero style)
# -----------------------
class MCTSNode:
    __slots__ = ('parent','prior','visit_count','value_sum','children','reward','to_play','action')
    def __init__(self, parent=None, prior=0.0, action=None):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode
        self.reward = 0.0   # reward received on edge to this node
        self.to_play = 0
        self.action = action

    def expanded(self):
        return len(self.children) > 0

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

def ucb_score(parent, child, c_puct=1.25, pb_c_base=19652, pb_c_init=1.25):
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    q = child.q_value()
    return q + prior_score

def run_mcts(root_state, repr_net, dyn_net, pred_net, n_sim=50, depth_limit=10, action_dim=2, gamma=0.99):
    # root_state: torch tensor (1, window, input_dim) on device
    with torch.no_grad():
        root_latent = repr_net(root_state)  # (1, hidden_dim)
        logits, value = pred_net(root_latent)
        priors = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        root = MCTSNode(parent=None, prior=1.0)
        # expand root children
        for a in range(action_dim):
            root.children[a] = MCTSNode(parent=root, prior=float(priors[a]), action=a)

    for _ in range(n_sim):
        node = root
        latent = root_latent.clone()
        search_path = [node]
        depth = 0
        # selection
        while node.expanded() and depth < depth_limit:
            # select child with max UCB
            best_a, best_child = max(node.children.items(), key=lambda kv: ucb_score(node, kv[1]))
            node = best_child
            search_path.append(node)
            # apply dynamics to latent to get next latent; for selection we need the latent corresponding to node
            a = node.action
            action_onehot = torch.zeros(1, action_dim, device=device)
            action_onehot[0, a] = 1.0
            with torch.no_grad():
                latent, reward = dyn_net(latent, action_onehot)
            depth += 1

        # expand leaf
        if not node.expanded():
            with torch.no_grad():
                logits, value = pred_net(latent)
                priors = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            for a in range(action_dim):
                node.children[a] = MCTSNode(parent=node, prior=float(priors[a]), action=a)
            node.reward = float(reward.cpu().item()) if 'reward' in locals() else 0.0
            leaf_value = float(value.cpu().item())
        else:
            # reached depth limit: bootstrap value
            with torch.no_grad():
                _, leaf_value_t = pred_net(latent)
                leaf_value = float(leaf_value_t.cpu().item())

        # backup
        v = leaf_value
        for nd in reversed(search_path):
            # if node has reward on edge to it, incorporate it
            if nd is not root:
                r = nd.reward
                v = r + gamma * v
            nd.visit_count += 1
            nd.value_sum += v

    # compute root visit distribution
    visits = np.array([root.children[a].visit_count for a in sorted(root.children.keys())], dtype=np.float32)
    pi = visits / (visits.sum() + 1e-8)
    root_value = root.q_value()
    return pi, root_value

# -----------------------
# Replay buffer (stores MCTS targets)
# -----------------------
ReplayEntry = collections.namedtuple('ReplayEntry', ['obs','action','reward','pi','z'])

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(ReplayEntry(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, min(len(self.buf), batch_size))
        return batch
    def __len__(self):
        return len(self.buf)

# -----------------------
# Training loop implementing MuZero self-play with MCTS
# -----------------------
def train_muzero_full(kraken: KrakenWrapper,
                      data_dir=None,
                      arch='lstm',
                      hidden_dim=64,
                      window=21,
                      action_dim=2,
                      epochs=100,
                      games_per_epoch=32,
                      max_steps=40,
                      n_sim=50,
                      depth_limit=10,
                      unroll_steps=5,
                      batch_size=64,
                      lr=1e-3,
                      gamma=0.99,
                      checkpoint_dir="./checkpoints_muzero"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # initialize networks
    # we need input_dim by sampling one data file (fallback)
    df_dict = {}
    if data_dir:
        # load files if present
        for f in os.listdir(data_dir):
            if f.endswith(".txt") or f.endswith(".csv"):
                try:
                    import pandas as pd
                    df_dict[f] = pd.read_csv(os.path.join(data_dir, f))
                except Exception:
                    pass

    # helper to obtain dataframe for an asset
    def get_df_for_asset(asset):
        # prefer local files in data_dir matching asset
        if data_dir:
            for k, df in df_dict.items():
                if asset.replace("/", "").replace("USDT","") in k or asset.replace("/","") in k:
                    return df
        # fallback: try pulling single symbol via kraken wrapper
        try:
            # wrapper expects pair like "ADA/USD" in some methods; attempt common formats
            cand = asset
            if "/" not in cand:
                cand = asset[:-4] + "/USD" if asset.endswith("USDT") else asset
            df = kraken.pull_single_sym_hist(cand, lb=kraken.look_back)
            return df
        except Exception:
            # last fallback: use any loaded file
            if len(df_dict) > 0:
                return next(iter(df_dict.values()))
            raise RuntimeError("No data available for asset " + str(asset))

    # get one sample df to infer input dim
    sample_df = None
    if len(df_dict) > 0:
        sample_df = next(iter(df_dict.values()))
    else:
        # try to fetch one asset name
        try:
            assets = kraken.get_usdt_assets()
            if len(assets) > 0:
                sample_df = get_df_for_asset(assets[0])
        except Exception:
            pass

    if sample_df is None:
        raise RuntimeError("No historical data available locally and kraken fetch failed.")

    # apply features once to sample to infer dim
    sample_df = apply_slope_features(sample_df, columns=['close','open','high','low'], dropna=True)
    # remove non-numeric cols like 'time' or 'date'
    sample_df = sample_df.drop(columns=[c for c in sample_df.columns if c.lower() in ('time','date')], errors='ignore')
    input_dim = sample_df.shape[1]

    repr_net = RepresentationNet(input_dim, arch=arch, hidden_dim=hidden_dim).to(device)
    dyn_net = DynamicsNet(state_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    pred_net = PredictionNet(state_dim=hidden_dim, action_dim=action_dim).to(device)

    params = list(repr_net.parameters()) + list(dyn_net.parameters()) + list(pred_net.parameters())
    opt = optim.Adam(params, lr=lr)
    replay = ReplayBuffer(capacity=20000)

    for epoch in range(epochs):
        # pick random asset from kraken's usdt list
        assets = kraken.get_usdt_assets()
        if len(assets) == 0:
            raise RuntimeError("Kraken returned no assets.")
        asset = random.choice(assets)
        try:
            df = get_df_for_asset(asset)
        except Exception:
            df = sample_df.copy()

        # prepare df and env
        try:
            df = apply_slope_features(df, columns=['close','open','high','low'], dropna=True)
        except Exception:
            pass
        # drop non-feature columns
        df = df.reset_index(drop=True)
        df = df.drop(columns=[c for c in df.columns if c.lower() in ('time','date')], errors='ignore')
        env = LSTMMarketEnv(df, window=window)

        # self-play games for this epoch
        for g in range(games_per_epoch):
            # random start ensuring room
            max_start = max(0, env.horizon - window - max_steps - 1)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            obs = env.reset(start_idx)
            traj = []  # list of (obs, action, reward, pi)
            done = False
            steps = 0
            while not done and steps < max_steps and env.t < env.horizon - 1:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, window, input_dim)
                pi, root_v = run_mcts(obs_tensor, repr_net, dyn_net, pred_net,
                                      n_sim=n_sim, depth_limit=depth_limit,
                                      action_dim=action_dim, gamma=gamma)
                action = int(np.random.choice(action_dim, p=pi))
                next_obs, reward, done, _ = env.step(action)
                traj.append((obs.copy(), action, reward, pi))
                obs = next_obs
                steps += 1
            # compute discounted returns for trajectory
            rewards = [t[2] for t in traj]
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.append(R)
            returns = list(reversed(returns))
            # push to replay: for each step store obs, action, reward, pi, z (return)
            for i, (obs_seq, action, reward, pi) in enumerate(traj):
                z = returns[i]
                replay.push(obs_seq, action, reward, pi, z)

        # training from replay
        if len(replay) >= 1:
            for _ in range(max(1, len(replay)//batch_size)):
                batch = replay.sample(batch_size)
                B = len(batch)
                obs_batch = torch.tensor([b.obs for b in batch], dtype=torch.float32).to(device)  # (B, window, input_dim)
                action_batch = torch.tensor([b.action for b in batch], dtype=torch.long).to(device)
                pi_batch = torch.tensor([b.pi for b in batch], dtype=torch.float32).to(device)  # (B, action_dim)
                z_batch = torch.tensor([b.z for b in batch], dtype=torch.float32).unsqueeze(1).to(device)

                # representation
                state = repr_net(obs_batch)  # (B, hidden_dim)
                logits, value = pred_net(state)
                policy_log_prob = nn.functional.log_softmax(logits, dim=-1)
                policy_loss = - (pi_batch * policy_log_prob).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(value, z_batch)

                # reward/prediction unroll losses
                consistency_loss = 0.0
                state_u = state
                for k in range(unroll_steps):
                    # use action_batch as proxy for unroll actions (simple approximation)
                    action_onehot = torch.zeros(B, action_dim, device=device)
                    action_onehot[range(B), action_batch] = 1.0
                    state_u, reward_pred = dyn_net(state_u, action_onehot)
                    logits_u, value_u = pred_net(state_u)
                    # use z_batch as weak supervision for unrolled values
                    consistency_loss = consistency_loss + nn.functional.mse_loss(value_u, z_batch)
                    # reward target: use zero or immediate reward from replay (simple)
                    # we don't have multi-step reward targets aligned; skip strict reward supervision here

                loss = policy_loss + value_loss + 0.5 * consistency_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        # checkpoint
        torch.save({
            'epoch': epoch,
            'repr_state': repr_net.state_dict(),
            'dyn_state': dyn_net.state_dict(),
            'pred_state': pred_net.state_dict(),
            'opt_state': opt.state_dict(),
        }, os.path.join(checkpoint_dir, f"muzero_{epoch}.pth"))

        print(f"Epoch {epoch} done. replay_size={len(replay)}")

    print("Training complete.")

# Example usage (uncomment to run as script)
if __name__ == "__main__":
    kr = KrakenWrapper()
    train_muzero_full(kr, data_dir="./hist_data/crypto/kraken_1day/",
                      arch='lstm', hidden_dim=64, window=21,
                      epochs=10, games_per_epoch=16, max_steps=20,
                      n_sim=30, depth_limit=8, unroll_steps=3, batch_size=64)
# filepath: c:\Users\nick5\dev\hypercube_ai\ai\treetrader\mu_zero_trader.py
import os
import random
import math
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from api_wrappers.kraken_wrapper import KrakenWrapper
from utils.feature_functions import apply_slope_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Architectures
# -----------------------
class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # (B, hidden_dim)

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        x = self.fc(x)
        attn_out, _ = self.attn(x, x, x)
        return attn_out[:, -1, :]

class ConvBlock(nn.Module):
    def __init__(self, input_dim, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, out_channels, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)

# -----------------------
# MuZero nets: representation, dynamics (with reward), prediction
# -----------------------
class RepresentationNet(nn.Module):
    def __init__(self, input_dim, arch='lstm', hidden_dim=64):
        super().__init__()
        if arch == 'lstm':
            self.block = LSTMBlock(input_dim, hidden_dim)
        elif arch == 'attention':
            self.block = AttentionBlock(input_dim, hidden_dim)
        elif arch == 'conv':
            self.block = ConvBlock(input_dim, out_channels=hidden_dim)
        else:
            raise ValueError("Unknown arch")
    def forward(self, obs):  # obs: (B, seq_len, input_dim)
        return self.block(obs)  # (B, hidden_dim)

class DynamicsNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
    def forward(self, state, action_onehot):  # state: (B, state_dim)
        x = torch.cat([state, action_onehot], dim=-1)
        h = self.fc1(x)
        next_s = self.next_state(h)
        reward = self.reward_head(h)
        return next_s, reward

class PredictionNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = nn.Linear(state_dim, action_dim)
        self.value = nn.Linear(state_dim, 1)
    def forward(self, state):
        return self.policy(state), self.value(state)

# -----------------------
# Environment wrapper (lightweight LSTMMarketEnv)
# -----------------------
class LSTMMarketEnv:
    def __init__(self, df, window=21, cost=0.0, initial_cash=1000.0, max_position=10):
        # df: pandas DataFrame with features, must include 'close' column
        self.data = df.reset_index(drop=True)
        self.horizon = len(self.data)
        self.window = window
        self.cost = cost
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.reset()

    def reset(self, start_idx=0):
        self.t = int(start_idx)
        self.position = 0
        self.cash = float(self.initial_cash)
        self.price = float(self.data.loc[self.t, 'close'])
        self.max_portfolio_value = self.cash
        return self._obs()

    def _obs(self):
        start = max(0, self.t - self.window + 1)
        rows = self.data.iloc[start:self.t+1].drop(columns=['date'], errors='ignore').values.astype(np.float32)
        if len(rows) < self.window:
            pad = np.zeros((self.window - len(rows), rows.shape[1]), dtype=np.float32)
            rows = np.vstack([pad, rows])
        return rows  # shape: (window, feature_dim)

    def step(self, action):
        # action: 0=buy, 1=sell
        prev_value = self.cash + self.position * self.price
        # buy only if enough cash and within position limit
        if action == 0 and self.cash >= self.price * (1 + self.cost) and self.position < self.max_position:
            self.position += 1
            self.cash -= self.price * (1 + self.cost)
        # sell only if position > 0
        elif action == 1 and self.position > 0:
            self.position -= 1
            self.cash += self.price * (1 - self.cost)
        # advance
        self.t += 1
        done = (self.t >= self.horizon)
        if not done:
            self.price = float(self.data.loc[self.t, 'close'])
        value = self.cash + self.position * self.price
        # immediate P&L reward (relative) and penalize drawdown
        reward = (value - prev_value) / (prev_value + 1e-8)
        self.max_portfolio_value = max(self.max_portfolio_value, value)
        drawdown = (self.max_portfolio_value - value) / (self.max_portfolio_value + 1e-8)
        reward = reward - 0.5 * drawdown  # tune weight
        return self._obs(), float(reward), done, {}

# -----------------------
# MCTS Implementation (MuZero style)
# -----------------------
class MCTSNode:
    __slots__ = ('parent','prior','visit_count','value_sum','children','reward','to_play','action')
    def __init__(self, parent=None, prior=0.0, action=None):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode
        self.reward = 0.0   # reward received on edge to this node
        self.to_play = 0
        self.action = action

    def expanded(self):
        return len(self.children) > 0

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

def ucb_score(parent, child, c_puct=1.25, pb_c_base=19652, pb_c_init=1.25):
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    q = child.q_value()
    return q + prior_score

def run_mcts(root_state, repr_net, dyn_net, pred_net, n_sim=50, depth_limit=10, action_dim=2, gamma=0.99):
    # root_state: torch tensor (1, window, input_dim) on device
    with torch.no_grad():
        root_latent = repr_net(root_state)  # (1, hidden_dim)
        logits, value = pred_net(root_latent)
        priors = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        root = MCTSNode(parent=None, prior=1.0)
        # expand root children
        for a in range(action_dim):
            root.children[a] = MCTSNode(parent=root, prior=float(priors[a]), action=a)

    for _ in range(n_sim):
        node = root
        latent = root_latent.clone()
        search_path = [node]
        depth = 0
        # selection
        while node.expanded() and depth < depth_limit:
            # select child with max UCB
            best_a, best_child = max(node.children.items(), key=lambda kv: ucb_score(node, kv[1]))
            node = best_child
            search_path.append(node)
            # apply dynamics to latent to get next latent; for selection we need the latent corresponding to node
            a = node.action
            action_onehot = torch.zeros(1, action_dim, device=device)
            action_onehot[0, a] = 1.0
            with torch.no_grad():
                latent, reward = dyn_net(latent, action_onehot)
            depth += 1

        # expand leaf
        if not node.expanded():
            with torch.no_grad():
                logits, value = pred_net(latent)
                priors = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            for a in range(action_dim):
                node.children[a] = MCTSNode(parent=node, prior=float(priors[a]), action=a)
            node.reward = float(reward.cpu().item()) if 'reward' in locals() else 0.0
            leaf_value = float(value.cpu().item())
        else:
            # reached depth limit: bootstrap value
            with torch.no_grad():
                _, leaf_value_t = pred_net(latent)
                leaf_value = float(leaf_value_t.cpu().item())

        # backup
        v = leaf_value
        for nd in reversed(search_path):
            # if node has reward on edge to it, incorporate it
            if nd is not root:
                r = nd.reward
                v = r + gamma * v
            nd.visit_count += 1
            nd.value_sum += v

    # compute root visit distribution
    visits = np.array([root.children[a].visit_count for a in sorted(root.children.keys())], dtype=np.float32)
    pi = visits / (visits.sum() + 1e-8)
    root_value = root.q_value()
    return pi, root_value

# -----------------------
# Replay buffer (stores MCTS targets)
# -----------------------
ReplayEntry = collections.namedtuple('ReplayEntry', ['obs','action','reward','pi','z'])

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(ReplayEntry(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, min(len(self.buf), batch_size))
        return batch
    def __len__(self):
        return len(self.buf)

# -----------------------
# Training loop implementing MuZero self-play with MCTS
# -----------------------
def train_muzero_full(kraken: KrakenWrapper,
                      data_dir=None,
                      arch='lstm',
                      hidden_dim=64,
                      window=21,
                      action_dim=2,
                      epochs=100,
                      games_per_epoch=32,
                      max_steps=40,
                      n_sim=50,
                      depth_limit=10,
                      unroll_steps=5,
                      batch_size=64,
                      lr=1e-3,
                      gamma=0.99,
                      checkpoint_dir="./checkpoints_muzero"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # initialize networks
    # we need input_dim by sampling one data file (fallback)
    df_dict = {}
    if data_dir:
        # load files if present
        for f in os.listdir(data_dir):
            if f.endswith(".txt") or f.endswith(".csv"):
                try:
                    import pandas as pd
                    df_dict[f] = pd.read_csv(os.path.join(data_dir, f))
                except Exception:
                    pass

    # helper to obtain dataframe for an asset
    def get_df_for_asset(asset):
        # prefer local files in data_dir matching asset
        if data_dir:
            for k, df in df_dict.items():
                if asset.replace("/", "").replace("USDT","") in k or asset.replace("/","") in k:
                    return df
        # fallback: try pulling single symbol via kraken wrapper
        try:
            # wrapper expects pair like "ADA/USD" in some methods; attempt common formats
            cand = asset
            if "/" not in cand:
                cand = asset[:-4] + "/USD" if asset.endswith("USDT") else asset
            df = kraken.pull_single_sym_hist(cand, lb=kraken.look_back)
            return df
        except Exception:
            # last fallback: use any loaded file
            if len(df_dict) > 0:
                return next(iter(df_dict.values()))
            raise RuntimeError("No data available for asset " + str(asset))

    # get one sample df to infer input dim
    sample_df = None
    if len(df_dict) > 0:
        sample_df = next(iter(df_dict.values()))
    else:
        # try to fetch one asset name
        try:
            assets = kraken.get_usdt_assets()
            if len(assets) > 0:
                sample_df = get_df_for_asset(assets[0])
        except Exception:
            pass

    if sample_df is None:
        raise RuntimeError("No historical data available locally and kraken fetch failed.")

    # apply features once to sample to infer dim
    sample_df = apply_slope_features(sample_df, columns=['close','open','high','low'], dropna=True)
    # remove non-numeric cols like 'time' or 'date'
    sample_df = sample_df.drop(columns=[c for c in sample_df.columns if c.lower() in ('time','date')], errors='ignore')
    input_dim = sample_df.shape[1]

    repr_net = RepresentationNet(input_dim, arch=arch, hidden_dim=hidden_dim).to(device)
    dyn_net = DynamicsNet(state_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    pred_net = PredictionNet(state_dim=hidden_dim, action_dim=action_dim).to(device)

    params = list(repr_net.parameters()) + list(dyn_net.parameters()) + list(pred_net.parameters())
    opt = optim.Adam(params, lr=lr)
    replay = ReplayBuffer(capacity=20000)

    for epoch in range(epochs):
        # pick random asset from kraken's usdt list
        assets = kraken.get_usdt_assets()
        if len(assets) == 0:
            raise RuntimeError("Kraken returned no assets.")
        asset = random.choice(assets)
        try:
            df = get_df_for_asset(asset)
        except Exception:
            df = sample_df.copy()

        # prepare df and env
        try:
            df = apply_slope_features(df, columns=['close','open','high','low'], dropna=True)
        except Exception:
            pass
        # drop non-feature columns
        df = df.reset_index(drop=True)
        df = df.drop(columns=[c for c in df.columns if c.lower() in ('time','date')], errors='ignore')
        env = LSTMMarketEnv(df, window=window)

        # self-play games for this epoch
        for g in range(games_per_epoch):
            # random start ensuring room
            max_start = max(0, env.horizon - window - max_steps - 1)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            obs = env.reset(start_idx)
            traj = []  # list of (obs, action, reward, pi)
            done = False
            steps = 0
            while not done and steps < max_steps and env.t < env.horizon - 1:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, window, input_dim)
                pi, root_v = run_mcts(obs_tensor, repr_net, dyn_net, pred_net,
                                      n_sim=n_sim, depth_limit=depth_limit,
                                      action_dim=action_dim, gamma=gamma)
                action = int(np.random.choice(action_dim, p=pi))
                next_obs, reward, done, _ = env.step(action)
                traj.append((obs.copy(), action, reward, pi))
                obs = next_obs
                steps += 1
            # compute discounted returns for trajectory
            rewards = [t[2] for t in traj]
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.append(R)
            returns = list(reversed(returns))
            # push to replay: for each step store obs, action, reward, pi, z (return)
            for i, (obs_seq, action, reward, pi) in enumerate(traj):
                z = returns[i]
                replay.push(obs_seq, action, reward, pi, z)

        # training from replay
        if len(replay) >= 1:
            for _ in range(max(1, len(replay)//batch_size)):
                batch = replay.sample(batch_size)
                B = len(batch)
                obs_batch = torch.tensor([b.obs for b in batch], dtype=torch.float32).to(device)  # (B, window, input_dim)
                action_batch = torch.tensor([b.action for b in batch], dtype=torch.long).to(device)
                pi_batch = torch.tensor([b.pi for b in batch], dtype=torch.float32).to(device)  # (B, action_dim)
                z_batch = torch.tensor([b.z for b in batch], dtype=torch.float32).unsqueeze(1).to(device)

                # representation
                state = repr_net(obs_batch)  # (B, hidden_dim)
                logits, value = pred_net(state)
                policy_log_prob = nn.functional.log_softmax(logits, dim=-1)
                policy_loss = - (pi_batch * policy_log_prob).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(value, z_batch)

                # reward/prediction unroll losses
                consistency_loss = 0.0
                state_u = state
                for k in range(unroll_steps):
                    # use action_batch as proxy for unroll actions (simple approximation)
                    action_onehot = torch.zeros(B, action_dim, device=device)
                    action_onehot[range(B), action_batch] = 1.0
                    state_u, reward_pred = dyn_net(state_u, action_onehot)
                    logits_u, value_u = pred_net(state_u)
                    # use z_batch as weak supervision for unrolled values
                    consistency_loss = consistency_loss + nn.functional.mse_loss(value_u, z_batch)
                    # reward target: use zero or immediate reward from replay (simple)
                    # we don't have multi-step reward targets aligned; skip strict reward supervision here

                loss = policy_loss + value_loss + 0.5 * consistency_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        # checkpoint
        torch.save({
            'epoch': epoch,
            'repr_state': repr_net.state_dict(),
            'dyn_state': dyn_net.state_dict(),
            'pred_state': pred_net.state_dict(),
            'opt_state': opt.state_dict(),
        }, os.path.join(checkpoint_dir, "muzero_latest.pth"))

        print(f"Epoch {epoch} done. replay_size={len(replay)}")

    print("Training complete.")

# Example usage (uncomment to run as script)
if __name__ == "__main__":
    kr = KrakenWrapper()
    train_muzero_full(kr, data_dir="./hist_data/crypto/kraken_1day/",
                      arch='lstm', hidden_dim=64, window=21,
                      epochs=10, games_per_epoch=16, max_steps=20,
                      n_sim=30, depth_limit=8, unroll_steps=3, batch_size=64)