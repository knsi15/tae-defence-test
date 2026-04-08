import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from scipy.optimize import differential_evolution

from lib.preprocess import load_data, load_model, save_org_data, save_ae_data
from tensorflow.keras.utils import to_categorical

# 評価関数の定義
def eval_misclassification(probs, perturbations, label, original_x, perturbed_x, lim):
    return -(1 - probs[0, label])

def eval_combined_normalized(probs, perturbations, label, original_x, perturbed_x, lim):
    alpha = perturbations[-1]  # DEで最適化されるα値 (0.0〜1.0)

    misclass_score = -(1 - probs[0, label])
    mae = np.mean(np.abs(perturbed_x - original_x))
    perturb_score = -max(0.0, 1 - (mae / lim))
    return alpha * misclass_score + (1-alpha) * perturb_score

def eval_misclass_with_lim(probs, perturbations, label, original_x, perturbed_x, lim):
    """
    誤分類 + lim最小化 の2項を組み合わせた評価関数
    誤分類を優先しつつ、limが小さいほど良い評価を与える
    """
    alpha = 0.7   # 誤分類の重み
    beta  = 0.3   # lim最小化の重み

    # 誤分類スコア: 元ラベルの信頼度が低いほど良い (負値、小さいほど良い)
    misclass_score = -(1 - probs[0, label])

    # limスコア: limが小さいほど良い (負値、小さいほど良い)
    lim_score = -(1 - (lim / LIM_MAX))

    return alpha * misclass_score + beta * lim_score

eval_funcs = {
    "misclassification": eval_misclassification,
    # "combined_normalized": eval_combined_normalized,
    # "misclass_with_lim": eval_misclass_with_lim
}

# 目的関数の定義
def objective_function_multiitem(perturbations, model, x, y, nb_classes):

    label = np.where(y > 0) # xのクラスラベルの取得
    x = np.copy(x)
    x = x.reshape(1, x.shape[0],x.shape[1])

    # 遺伝子型を用いて、敵対的サンプルを生成
    plist = [perturbations[i:i+2] for i in range(0, len(perturbations), 2)]
    for p in plist:
        pos = int(p[0])
        val = p[1]
        x[0, pos] = x[0, pos] + val

    # モデルを通して分類確率を取得
    prep = model(x)

    return -(1-prep.numpy()[0, label]) #誤分類確率を負に変換、値が小さいほど良い個体


def main(dataset, model_type):
    
    x_test, y_test = load_data(dataset, is_test=True) # データの読み出し
    
    nb_classes = len(np.unique(y_test)) #クラス数
    y_true = y_test
    y_test = to_categorical(y_test)
    tlen = len(x_test[0]) # 時系列長

    # 遺伝子型の定義[（位置、摂動）, （位置、摂動）, （位置、摂動）...] 
    lim = 0.3
    perturbation = [(0, tlen-1), (-lim, lim), (0, tlen-1), (-lim, lim), (0, tlen-1), (-lim, lim),(0, tlen-1), (-lim, lim), (0, tlen-1), (-lim, lim)]

    # モデルのロード
    model = load_model(dataset, model_type)

    # 摂動を加える前のデータ保存（標準化済み）
    save_org_data(dataset, model_type,x_test)

    # 摂動を加える前の精度計測
    pred = model(x_test)  
    test_acc = tf.metrics.SparseCategoricalAccuracy()
    test_acc(y_true, pred)
    org_acc = test_acc.result().numpy()*100

    # テストデータ１つづつAEを作成
    for index in range(len(x_test)):
        result = differential_evolution(
            func = objective_function_multiitem,
            bounds = perturbation,
            args = (model, x_test[index], y_test[index], nb_classes),  
            strategy='rand1exp', # One of the common DE strategies
            maxiter=200,         # Max number of generations
            popsize=30,          # Population size multiplier (pop_size = popsize * num_dimensions)
            mutation=(0.5, 1.0), # F range
            recombination=0.7,   # CR
            disp=True,           # Display optimization progress
            #polish=True,         # Run local optimization at the end (optional)
            seed=42              # For reproducibility
        )

        best_perturbations = result.x
        best_negative_accuracy = result.fun

        # 最良個体の結果を時系列データに反映
        bplist = [best_perturbations[i:i+2] for i in range(0, len(best_perturbations), 2)]
        for p in bplist:
            pos = int(p[0])
            val = p[1]
            x_test[index][int(pos)][0][0] = x_test[index][int(pos)][0][0] + val

    # AEを保存
    save_ae_data(dataset, model_type, x_test)

    # 摂動を加えた後の精度を計測
    pred = model(x_test)
    test_acc = tf.metrics.SparseCategoricalAccuracy()
    test_acc(y_true,pred)
    ae_acc = test_acc.result().numpy()*100
    print(dataset+","+str(org_acc)+","+str(ae_acc))

if __name__ == "__main__":
    main("BeetleFly", "fcn")
 