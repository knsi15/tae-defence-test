import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.preprocess import load_data
from lib.autoencoder import TSAutoencoder
from lib.output_dirs import get_phase_dir, get_dataset_dir


def train_autoencoder(dataset,
                      output_dir,
                      epochs=300,
                      lr=1e-3,
                      batch_size=16,
                      val_ratio=0.2,
                      patience=20,
                      save_dir="models/autoencoder"):

    # 出力先準備
    phase_dir = get_phase_dir(output_dir, "train_ae")
    ds_dir = get_dataset_dir(phase_dir, dataset)

    # 1. データ読み込み (標準化済み)
    x_train, _ = load_data(dataset, is_test=False, norm=True)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    T = x_train.shape[1]

    # 2. train/val 分割 (シードを固定して再現性を確保)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_train))
    n_val = max(1, int(len(x_train) * val_ratio))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    x_tr = torch.tensor(x_train[tr_idx])
    x_val = torch.tensor(x_train[val_idx])

    # 3. モデル・オプティマイザ
    model = TSAutoencoder(T=T)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 学習ループ + Early Stopping
    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []

    start = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        epoch_losses = []
        for i in range(0, len(x_tr), batch_size):
            xb = x_tr[perm[i:i + batch_size]]
            out = model(xb)
            loss = criterion(out, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), x_val).item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{dataset}] Early stop at epoch {epoch + 1}")
                break

    duration = time.time() - start
    epochs_run = len(history)

    # 5. 履歴 CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(ds_dir, "loss_history.csv"), index=False)

    # 6. 学習曲線プロット
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"AE Training - {dataset}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ds_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 7. config.json
    config = {
        "dataset": dataset,
        "T": int(T),
        "n_train_total": int(len(x_train)),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "epochs_max": epochs,
        "epochs_run": epochs_run,
        "lr": lr,
        "batch_size": batch_size,
        "val_ratio": val_ratio,
        "patience": patience,
        "best_val_loss": float(best_val),
        "duration_sec": float(duration),
    }
    with open(os.path.join(ds_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 8. サマリ追記
    summary_path = os.path.join(phase_dir, "train_summary.xlsx")
    row_df = pd.DataFrame([config])
    if os.path.exists(summary_path):
        existing = pd.read_excel(summary_path)
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_excel(summary_path, index=False)

    # 9. モデル保存
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, f"ae_{dataset}.pth"))
    print(f"[{dataset}] saved. epochs_run={epochs_run}, best_val_loss={best_val:.6f}")
