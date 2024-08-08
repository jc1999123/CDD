import numpy as np
import torch
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
from model import MLPDiffusion  # Import the model from model.py

def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]
    e = torch.randn_like(x_0)
    x = x_0 * a + e * aml
    output = model(x, t.squeeze(-1))

    return (e - output).square().mean()

def train_model(expression, num_steps=200, num_epoch=4000, batch_size=2, k_fold=3):
    dataset = torch.Tensor(expression.values).float()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    x_now = np.array(expression.values)
    kf = KFold(n_splits=k_fold)
    kk = 0

    for train_index, test_index in kf.split(x_now):
        all_train, all_test = x_now[train_index], x_now[test_index]
        dataset_train = torch.Tensor(all_train).float()

        model = MLPDiffusion(num_steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for t in range(num_epoch):
            for idx, batch_x in enumerate(dataloader):
                loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

            if t % 4000 == 0:
                print(loss)

        torch.save(model, "model/clu1" + str(kk) + "_model.h5")
        kk += 1

if __name__ == "__main__":
    expression = pd.read_table('clu1.csv', sep=',')
    expression = expression.iloc[:, 1:]
    print("shape of s:", np.shape(expression))
    train_model(expression)
