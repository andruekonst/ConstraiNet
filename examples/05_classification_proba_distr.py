from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from constrainet import (
    LinearConstraints,
    LinearEqualityConstraints,
    QuadraticConstraints,
    ConstraiNetLayer,
    DenseConstraiNet
)
from constrainet.models.constrainet import make_constraint_set, make_feed_forward_network
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
import pandas as pd
import seaborn as sns


results_path = Path(__file__).parent.resolve() / '05_classification_results'
results_path.mkdir(parents=True, exist_ok=True)


LossKind = Literal['CEL', 'NLL', 'HINGE']


def optimize_classifier(nn: torch.nn.Module,
                       n_epochs: int,
                       X_train,
                       y_train,
                       X_val=None,
                       y_val=None,
                       n_features: int = None,
                       batch_size: int = 100,
                       lr: float = 1.e-2,
                       reduce_lr_factor: float = 0.9,
                       reduce_lr_patience: int = 20,
                       reduce_lr_cooldown: int = 20,
                       loss_kind: LossKind = 'NLL'):
    X_train = torch.tensor(X_train, dtype=torch.double)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.double)
        y_val = torch.tensor(y_val, dtype=torch.long)
        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_dataloader = None

    if loss_kind == 'CEL':
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        process_out_fn = lambda x: x
    elif loss_kind == 'NLL':
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        process_out_fn = lambda x: torch.log(torch.clip(x, EPS, 1.0 - EPS))
    elif loss_kind == 'HINGE':
        loss_fn = torch.nn.MultiMarginLoss(reduction='sum')
        process_out_fn = lambda x: x
    else:
        raise ValueError(f'{loss_kind=!r}')

    optim = torch.optim.Adam(nn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        cooldown=reduce_lr_cooldown,
        mode='min',
        min_lr=1.e-5
    )

    history = []
    val_history = []

    EPS = 1.e-6

    for epoch in range(n_epochs):
        nn.train()
        train_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            pred_solutions = process_out_fn(nn(batch_X))
            loss = loss_fn(pred_solutions, batch_y) / len(batch_X)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(batch_X)
        train_loss /= len(train_dataset)
        history.append(train_loss)
        scheduler.step(train_loss)

        # validation
        if val_dataloader is not None:
            nn.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_X, batch_y in val_dataloader:
                    val_loss += loss_fn(process_out_fn(nn(batch_X)), batch_y).item()
                val_loss /= len(val_dataset)
                val_history.append(val_loss)

    return history, val_history


def pretrain_uniform(nn: torch.nn.Module,
                     n_epochs: int,
                     X_train,
                     y_train,
                     batch_size: int = 100,
                     lr: float = 1.e-2):
    X_train = torch.tensor(X_train, dtype=torch.double)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(nn.parameters(), lr=lr)

    history = []

    for epoch in range(n_epochs):
        nn.train()
        train_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            loss = ((nn(batch_X)[:, 0]) ** 2.0).sum() / len(batch_X)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(batch_X)
        train_loss /= len(train_dataset)
        history.append(train_loss)
    return history


def make_softmax_network(input_dim: int, hidden_dim: int = 20, out_dim: int = 10,
                         n_layers: int = 5,
                         activation=torch.nn.ReLU):
    return torch.nn.Sequential(
        make_feed_forward_network(input_dim, hidden_dim, out_dim, n_layers, activation),
        torch.nn.Softmax(dim=1)
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Feel free to change parameters.
    """
    setting: Literal['constraints', 'projection', 'softmax'] = 'projection'
    loss_kind: LossKind ='NLL'
    nn_hidden_dim: int = 300
    nn_n_layers: int = 5
    random_state: int = 12345


def test_classification_problem(config: ExperimentConfig):
    torch.random.manual_seed(config.random_state)

    # all_X, all_y = load_iris(return_X_y=True)
    all_X, all_y = fetch_olivetti_faces(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        all_X, all_y,
        test_size=0.2,
        random_state=12345,
        stratify=all_y,
    )

    n_classes = len(np.unique(all_y))
    # discrete probability distribution constraints:
    eq_constraint = LinearEqualityConstraints(
        A=np.ones((1, n_classes), dtype=np.float64),
        b=np.ones((1,), dtype=np.float64),
    )
    # centering equality constraint
    eq_constraint.null_space = (eq_constraint.null_space[0], np.full((n_classes,), 1 / n_classes, dtype=np.float64))
    constraints = make_constraint_set([
        LinearConstraints(
            A=-np.eye(n_classes, dtype=np.float64),
            b=np.zeros((n_classes,), dtype=np.float64),
        ),
        eq_constraint
    ])
    # centering pivot
    constraints.interior_point = np.zeros_like(constraints.interior_point)

    if config.setting != 'softmax':
        mode = 'center_projection' if config.setting == 'projection' else 'ray_shift'
        nn = DenseConstraiNet(
            input_dim=all_X.shape[1],
            hidden_dim=config.nn_hidden_dim,
            n_layers=config.nn_n_layers,
            constraints=constraints,
            mode=mode,
        )
        if config.setting == 'constraints':
            tmp_losses = pretrain_uniform(nn.dense, 1000, X_train, y_train, batch_size=200, lr=1.e-4)
            print("Pretrain loss from:", tmp_losses[0], "to:", tmp_losses[-1])
    else:
        nn = make_softmax_network(
            input_dim=all_X.shape[1],
            hidden_dim=config.nn_hidden_dim,
            out_dim=n_classes,
            n_layers=config.nn_n_layers,
            activation=torch.nn.ReLU,
        )

    experiment_name = f'{config.setting}_{config.loss_kind}'
    model_file = results_path / f'{experiment_name}_model.pth'
    loss_file = results_path / f'{experiment_name}_loss'
    if not model_file.exists():
        optim_history, test_history = optimize_classifier(
            nn,
            n_epochs=5000,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            n_features=n_classes,
            batch_size=200,
            lr=1.e-4,
            reduce_lr_patience=5000,
            loss_kind=config.loss_kind,
        )
        print("Saving model...")
        torch.save(nn.state_dict(), model_file)
        plt.figure()
        plt.plot(optim_history, label='train')
        plt.plot(test_history, label='test')
        plt.ylim([-0.1, 4.0])
        plt.legend()
        plt.savefig(f'{loss_file}.png', dpi=200)
        np.savez(
            f'{loss_file}.npz',
            optim_history=optim_history,
            test_history=test_history,
        )
        # plt.show()
    else:
        print("Loading model...")
        nn.load_state_dict(torch.load(model_file))

    with torch.no_grad():
        proj_simplex_points = nn(torch.tensor(X_test, dtype=torch.double)).numpy()

    print("All non-negative:", np.all(proj_simplex_points >= -1.e-9))
    print("Sum == 1:", np.allclose(proj_simplex_points.sum(axis=1), 1.0))

    print("Test  ROC-AUC:", roc_auc_score(y_test, proj_simplex_points, average='micro', multi_class='ovr'))
    print("Test       F1:", f1_score(y_test, np.argmax(proj_simplex_points, axis=1), average='micro'))
    print("Test Accuracy:", accuracy_score(y_test, np.argmax(proj_simplex_points, axis=1)))

    print("... On train ...")
    with torch.no_grad():
        proj_simplex_points = nn(torch.tensor(X_train, dtype=torch.double)).numpy()

    print("All non-negative:", np.all(proj_simplex_points >= -1.e-9))
    print("Sum == 1:", np.allclose(proj_simplex_points.sum(axis=1), 1.0))

    print("Train  ROC-AUC:", roc_auc_score(y_train, proj_simplex_points, average='micro', multi_class='ovr'))
    print("Train       F1:", f1_score(y_train, np.argmax(proj_simplex_points, axis=1), average='micro'))
    print("Train Accuracy:", accuracy_score(y_train, np.argmax(proj_simplex_points, axis=1)))


def plot_loss_comparison(loss_kind: LossKind):
    loss_files = {
        str(results_path / f'{setting}_{loss_kind}_loss.npz'): setting.capitalize()
        for setting in ['constraints', 'softmax', 'projection']
    }
    all_data = {
        title: np.load(results_path / fname)
        for fname, title in loss_files.items()
    }
    df = pd.concat(
        [
            pd.DataFrame({
                'train': data['optim_history'],
                'val': data['test_history'],
                'kind': [title] * len(data['optim_history']),
                'epoch': list(range(len(data['optim_history']))),
            })
            for title, data in all_data.items()
        ],
        axis=0
    )
    melt_df = pd.melt(
        df,
        id_vars=['kind', 'epoch'],
        value_vars=['train', 'val'],
        var_name='part',
        value_name='loss',
    )

    sns.relplot(
        data=melt_df,
        x='epoch',
        y='loss',
        hue='kind',
        kind='line',
        col='part',
    )
    plt.show()


if __name__ == '__main__':
    test_classification_problem(ExperimentConfig(
        setting='constraints',
        loss_kind='NLL',
    ))
    test_classification_problem(ExperimentConfig(
        setting='projection',
        loss_kind='NLL',
    ))
    test_classification_problem(ExperimentConfig(
        setting='softmax',
        loss_kind='NLL',
    ))
    plot_loss_comparison(loss_kind='NLL')
