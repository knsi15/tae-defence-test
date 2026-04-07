"""
実験結果のログ保存を担当するモジュール。
- 結果サマリの Excel 保存
- サンプル単位の詳細ログ保存
- 世代ごとの評価値ログ保存
"""

import os
import datetime
import pandas as pd


def save_result_to_excel(
    filename: str,
    dataset_name: str,
    model: str,
    eval_func_name: str,
    strategy: str,
    original_accuracy: float,
    attack_accuracy: float,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    gen: int,
    mae: float,
    mse: float,
    rmse: float,
    lim: float,
    avg_num_perturbations: float = None,
    max_perturbations: int = None,
    output_dir: str = None,
):
    """
    実験結果サマリを Excel に追記保存する。
    output_dir が指定されていれば <output_dir>/results_lab.xlsx に保存する。
    """
    duration = (end_time - start_time).total_seconds()
    minutes = int(duration / 60)
    result_row = {
        "開始日時": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "終了日時": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "実行時間（秒）": duration,
        "実行時間（分）": minutes,
        "平均世代数": gen,
        "モデル名": model,
        "データセット名": dataset_name,
        "戦略": strategy,
        "Lim": lim,
        "最大摂動数": max_perturbations,
        "平均摂動数": avg_num_perturbations,
        "評価関数": eval_func_name,
        "元精度": original_accuracy,
        "攻撃後精度": attack_accuracy,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }

    if output_dir:
        filename = os.path.join(output_dir, "results_lab.xlsx")

    if os.path.exists(filename):
        df = pd.read_excel(filename)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    df.to_excel(filename, index=False)


def save_detailed_results(detailed_results, output_dir, dataset, model_type, strategy, func, lim):
    """
    サンプルごとの詳細な実行結果をログ保存する。
    """
    if not detailed_results:
        return

    filename = f"detailed_log_{dataset}_{model_type}_{strategy}_{func}_lim{lim}.xlsx"
    filepath = os.path.join(output_dir, filename)

    pd.DataFrame(detailed_results).to_excel(filepath, index=False)
    print(f"詳細ログ保存完了: {filepath}")


def save_generation_log(generation_logs, output_dir, dataset, model_type, strategy, func, lim):
    """
    世代ごとの評価値を CSV 保存する。
    expected keys: sample_id, generation, best_value
    """
    if not generation_logs:
        return

    filename = f"generation_log_{dataset}_{model_type}_{strategy}_{func}_lim{lim}.csv"
    filepath = os.path.join(output_dir, filename)

    pd.DataFrame(generation_logs).to_csv(filepath, index=False)
    print(f"世代ログ保存完了: {filepath}")
