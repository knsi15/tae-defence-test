# tae-defence-test

時系列分類モデルに対する敵対的サンプル（Adversarial Example, AE）の生成と、Autoencoder を用いた検知手法の実験用リポジトリ。

## 概要

UCR 時系列データセット上で以下の3フェーズを実行する:

1. **attack** — 差分進化 (Differential Evolution) による敵対的サンプルの生成
2. **train** — クリーンデータで Autoencoder を学習
3. **eval** — 再構成誤差を用いて AE を検知し、TPR / FPR を算出

対象データセット: `BeetleFly`, `Car`, `Coffee`, `Computer`, `ECG200`, `ShapeletSim`, `ToeSegmentation2`
モデルタイプ: `fcn`

## セットアップ

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使い方

全フェーズを実行:

```bash
python main.py --phase all
```

個別に実行:

```bash
python main.py --phase attack
python main.py --phase train
python main.py --phase eval
```

既存の出力ディレクトリを再利用する場合:

```bash
python main.py --phase eval --output-dir <既存のrunディレクトリ>
```

## ディレクトリ構成

```
.
├── main.py                   # エントリポイント (attack/train/eval を順次実行)
├── mdeattack_timeseries.py   # DE による AE 生成のスタンドアロン実装
├── lib/                      # 攻撃・学習・検知・前処理モジュール
├── data/                     # データセット (gitignore 対象)
├── models/                   # 学習済みモデル (gitignore 対象)
└── requirements.txt
```

## 主な依存ライブラリ

torch / tensorflow / keras / aeon / scikit-learn / scipy / numpy / pandas / matplotlib
