import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.preprocess import load_data
from lib.autoencoder import TSAutoencoder
from lib.output_dirs import get_phase_dir, get_dataset_dir


def _load_ae(dataset, T, save_dir="models/autoencoder"):
    model = TSAutoencoder(T=T)
    model.load_state_dict(torch.load(os.path.join(save_dir, f"ae_{dataset}.pth")))
    model.eval()
    return model


def _reconstruction_errors(model, x):
    """x: numpy (N, T) → errors: numpy (N,)"""
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        out = model(xt)
        err = ((xt - out) ** 2).mean(dim=1).numpy()
    return err


def _reconstruct(model, x):
    """x: numpy (N, T) → reconstruction: numpy (N, T)"""
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        out = model(xt).numpy()
    return out


def _load_tsv(path):
    """save_org_data/save_ae_data の出力を読み込み (N, T) を返す"""
    return np.loadtxt(path, delimiter="\t").astype(np.float32)


def evaluate_detection(dataset,
                       model_type,
                       output_dir,
                       percentile=95):

    # 出力先準備
    phase_dir = get_phase_dir(output_dir, "detect")
    ds_dir = get_dataset_dir(phase_dir, dataset)

    # 1. train データから閾値を決める
    x_train, _ = load_data(dataset, is_test=False, norm=True)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    T = x_train.shape[1]

    model = _load_ae(dataset, T)
    train_errors = _reconstruction_errors(model, x_train)
    threshold = float(np.percentile(train_errors, percentile))

    # 2. 元 test データで FPR
    org_path = f"data/{dataset}/{dataset}_{model_type}_TEST_ORG.tsv"
    x_org = _load_tsv(org_path)
    org_errors = _reconstruction_errors(model, x_org)
    fpr = float((org_errors > threshold).mean())

    # 3. AE データで TPR
    ae_path = f"data/{dataset}/{dataset}_{model_type}_TEST_AE.tsv"
    x_ae = _load_tsv(ae_path)
    ae_errors = _reconstruction_errors(model, x_ae)
    tpr = float((ae_errors > threshold).mean())

    # 4. per-sample errors を long format で CSV
    err_df = pd.concat([
        pd.DataFrame({
            "sample_id": np.arange(len(train_errors)),
            "source": "train",
            "error": train_errors,
        }),
        pd.DataFrame({
            "sample_id": np.arange(len(org_errors)),
            "source": "test_org",
            "error": org_errors,
        }),
        pd.DataFrame({
            "sample_id": np.arange(len(ae_errors)),
            "source": "test_ae",
            "error": ae_errors,
        }),
    ], ignore_index=True)
    err_df.to_csv(os.path.join(ds_dir, "errors.csv"), index=False)

    # 5. 誤差分布ヒストグラム
    plt.figure(figsize=(9, 5))
    bins = 30
    plt.hist(train_errors, bins=bins, alpha=0.5, label="train", color="gray")
    plt.hist(org_errors, bins=bins, alpha=0.5, label="test_org", color="blue")
    plt.hist(ae_errors, bins=bins, alpha=0.5, label="test_ae", color="red")
    plt.axvline(threshold, color="black", linestyle="--",
                label=f"threshold (p{percentile})")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title(f"{dataset}: TPR={tpr:.3f}, FPR={fpr:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ds_dir, "hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 6. 再構成サンプルプロット (元/AE 各5件)
    n_show = min(5, len(x_org), len(x_ae))
    if n_show > 0:
        recon_org = _reconstruct(model, x_org[:n_show])
        recon_ae = _reconstruct(model, x_ae[:n_show])

        fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 5),
                                 squeeze=False)
        for i in range(n_show):
            axes[0, i].plot(x_org[i], color="blue", label="orig")
            axes[0, i].plot(recon_org[i], color="orange", label="recon", alpha=0.7)
            axes[0, i].set_title(f"org #{i}\nerr={org_errors[i]:.4f}")
            axes[0, i].grid(alpha=0.3)

            axes[1, i].plot(x_ae[i], color="red", label="AE")
            axes[1, i].plot(recon_ae[i], color="orange", label="recon", alpha=0.7)
            axes[1, i].set_title(f"AE #{i}\nerr={ae_errors[i]:.4f}")
            axes[1, i].grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(ds_dir, "recon_samples.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # 7. サマリ追記
    summary_path = os.path.join(phase_dir, "detection_summary.xlsx")
    row_df = pd.DataFrame([{
        "dataset": dataset,
        "model_type": model_type,
        "percentile": percentile,
        "threshold": threshold,
        "n_train": int(len(train_errors)),
        "n_test_org": int(len(org_errors)),
        "n_test_ae": int(len(ae_errors)),
        "TPR": tpr,
        "FPR": fpr,
    }])
    if os.path.exists(summary_path):
        existing = pd.read_excel(summary_path)
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_excel(summary_path, index=False)

    return tpr, fpr
