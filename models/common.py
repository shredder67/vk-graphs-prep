import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.data
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix, subgraph
import numpy as np

from tqdm import tqdm
from torchmetrics import Accuracy, Recall, Precision, F1Score
import matplotlib.pyplot as plt
import networkx as nx

from typing import List

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
    f1score = F1Score().to(device) 
    loss_hist = []
    f1_hist = []
    f1_val_hist = []
    with tqdm(range(1, epochs + 1)) as tepoch:
        for e in tepoch:
            optimizer.zero_grad()
            out = model(data).squeeze(-1)
            preds = torch.sigmoid(out)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            f1_hist.append(f1score(preds[data.train_mask], data.y[data.train_mask].to(torch.int)).item())

            if e%eval_freq == 0:
                f1_val_hist.append(
                    eval_model(model, data, {"acc": f1score}, device=device)["acc"]
                )
                if log_stats:
                    tepoch.write(f'\nepoch {e}:\ntr_loss {loss_hist[-1]:.4f}\ntr_f1: {f1_hist[-1]:.4f} \
                                    \nval_f1: {f1_val_hist[-1]:.4f}')

    return loss_hist, f1_hist, f1_val_hist


@torch.no_grad()
def eval_model(model, data, metrics: dict, use_val_mask=True, device=torch.device('cuda')):
    model.eval()
    mask = data.val_mask if use_val_mask else data.test_mask
    out = model(data).squeeze(-1)
    preds = torch.sigmoid(out)
    y_true = data.y[mask].to(torch.int)
    metric_values = dict()
    for metric_key, metric in metrics.items():
        metric_values[metric_key] = metric(preds[mask], y_true).item()
    return metric_values


def plot_hist(train_loss_hist, train_acc_hist, eval_acc_hist, eval_freq):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(train_loss_hist, label='train loss')
    ax1.set_title('Training Loss')
    ax2.plot(train_acc_hist, label='train f1-score')
    ax2.plot(list(range(0, len(train_acc_hist), eval_freq)), eval_acc_hist, label='eval f1-score')
    ax2.set_title('Training and Validation Metric (F1-Score)')
    plt.legend()
    plt.show()


def benchmark_model(model, data, apply_val_mask=False, apply_test_mask=True, metrics=None, device=torch.device('cpu')):
    if apply_test_mask == apply_val_mask:
        raise ValueError("Set val_mask or test_mask!")

    if metrics is None:
        metrics = {
            'recall': Recall().to(device),
            'precision': Precision().to(device),
            'f1-score': F1Score().to(device) 
        }
    
    return eval_model(model, data, metrics, apply_val_mask, device)
    

def plot_graph(data: torch_geometric.data.Data):
    """
    Отображает графическую структуру графа, rank-plot (отсортированные значения) степеней вершин, распределение степеней вершин
    """
    G = to_networkx(data, to_undirected=True)
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()


def plot_graph_adj_matrix(data: torch_geometric.data.Data):
    edge_sparse_matrix = to_scipy_sparse_matrix(edge_index=data.edge_index, edge_attr=data.edge_attr)
    plt.spy(edge_sparse_matrix)


def plot_subgraph(data: torch_geometric.data.Data, vert_list: List[int]):
    subgr_edge_index, subgr_edge_attr = subgraph(vert_list, data.edge_index, data.edge_attr)
    subgraph_data = torch_geometric.data.Data(edge_index=subgr_edge_index, edge_attr=subgr_edge_attr)
    plot_graph(subgraph_data)


def plot_features_distributions(data: torch_geometric.data.Data, vert_list: List[int], plot_type='box'):
    n = int(len(vert_list)**0.5)
    k = len(vert_list) // n
    if n*k < len(vert_list):
        k += 1
    fig, axs = plt.subplots(n, k)
    for i in range(n):
        for j in range(k):
            if i*k + j == len(vert_list):
                break
            if plot_type == 'hist':
                axs[i, j].hist(data.x[:, vert_list[i*k + j]])
            elif plot_type == 'violin':
                axs[i, j].violinplot(data.x[:, vert_list[i*k + j]])
            elif plot_type == 'box':
                axs[i, j].boxplot(data.x[:, vert_list[i*k + j]])

    plt.tight_layout()
    plt.show()

