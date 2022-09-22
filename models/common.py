import torch
import torch.nn as nn
import torch_geometric

from tqdm import tqdm
from torchmetrics import Accuracy
import matplotlib.pyplot as plt


def train_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: torch_geometric.data.Dataset,
        epochs=1000,
        eval_freq=100,
        log_stats=True,
        device=torch.device('cuda')):
        
    model.train()
    accuracy = Accuracy().to(device)
    loss_hist = []
    acc_hist = []
    acc_val_hist = []
    with tqdm(range(epochs)) as tepoch:
        for e in tepoch:
            optimizer.zero_grad()
            out = model(data).squeeze(-1)
            preds = torch.zeros(size=out.shape, dtype=torch.int, device=device)
            preds[out > 0.5] = 1
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            acc_hist.append(accuracy(preds[data.train_mask], data.y[data.train_mask].to(torch.int)).item())

            if log_stats and e%eval_freq == 0:
                acc_val_hist.append(
                    eval_model(model, data, {"acc": accuracy}, device=device)["acc"]
                )
                tepoch.write(f'\nepoch {e}:\ntr_loss {loss_hist[-1]:.4f}\ntr_acc: {acc_hist[-1]:.4f} \
                                \nval_acc: {acc_val_hist[-1]:.4f}')

    return loss_hist, acc_hist, acc_val_hist
    

@torch.no_grad()
def eval_model(model, data, metrics: dict, use_val_mask=True, device=torch.device('cuda')):
    model.eval()
    mask = data.val_mask if use_val_mask else data.test_mask
    out = model(data).squeeze(-1)
    preds = torch.zeros(size=out.shape, dtype=torch.int, device=device)
    preds[out > 0.5] = 1
    metric_values = dict.fromkeys(metrics.keys())
    for metric_key, metric in metrics.items():
        metric_values[metric_key] = metric(preds[mask], data.y[mask].to(torch.int)).item()
    return metric_values


def plot_hist(train_loss_hist, train_acc_hist, eval_acc_hist, eval_freq):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(train_loss_hist, label='train loss')
    ax2.plot(train_acc_hist, label='train accuracy')
    ax2.plot(range(0, len(train_acc_hist), eval_freq), eval_acc_hist, label='eval accuracy')
    plt.suptitle('Training process on github data')
    plt.legend()
    plt.show()