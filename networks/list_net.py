import torch
import torch.nn as nn
import torch.nn.functional as F

class ListNetRanker(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, X):
        # X: (batch, list_size, n_features)
        scores = self.net(X)  # (batch, list_size, 1)
        return scores.squeeze(-1)  # (batch, list_size)

def listnet_loss(scores, true_returns, temperature=0.01):
    # scores, true_returns: (batch, list_size)
    P_true = F.softmax(true_returns / temperature, dim=1)
    P_pred = F.softmax(scores, dim=1)
    loss = -(P_true * torch.log(P_pred + 1e-8)).sum(dim=1).mean()
    return loss

def train_listnet(model, dataloader, n_epochs=20, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        losses = []
        for X, y in dataloader:   # X: (batch, list_size, n_features)
            opt.zero_grad()
            scores = model(X)
            loss = listnet_loss(scores, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch}: mean loss = {np.mean(losses):.4f}")

def rank_assets(model, features_df, date):
    subset = features_df[features_df.date == date]
    X = torch.tensor(subset[feature_cols].values, dtype=torch.float32).unsqueeze(0)
    scores = model(X).detach().numpy()[0]
    subset['score'] = scores
    return subset.sort_values('score', ascending=False)

def tournament_rank(model, features_df, date, group_size=10, top_k=2):
    current = features_df[features_df.date == date]
    while len(current) > group_size:
        groups = [current.iloc[i:i+group_size] for i in range(0, len(current), group_size)]
        winners = []
        for g in groups:
            ranked = rank_assets(model, g, date)
            winners.append(ranked.head(top_k))
        current = pd.concat(winners)
    final_rank = rank_assets(model, current, date)
    return final_rank