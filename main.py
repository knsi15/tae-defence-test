import argparse

from lib.attack import generate_adversarial_examples
from lib.train_ae import train_autoencoder
from lib.detect import evaluate_detection
from lib.output_dirs import create_run_dir

DATASETS = ["BeetleFly", "Car", "Coffee", "Computers",
            "ECG200", "ShapeletSim", "ToeSegmentation2"]
MODEL_TYPE = "fcn"


def run_attack(output_dir):
    for d in DATASETS:
        try:
            generate_adversarial_examples(d, MODEL_TYPE)
        except Exception as e:
            print(f"[attack] {d} failed: {e}")


def run_train(output_dir):
    for d in DATASETS:
        try:
            train_autoencoder(d, output_dir=output_dir)
        except Exception as e:
            print(f"[train] {d} failed: {e}")


def run_eval(output_dir):
    results = []
    for d in DATASETS:
        try:
            tpr, fpr = evaluate_detection(d, MODEL_TYPE, output_dir=output_dir)
            results.append((d, tpr, fpr))
            print(f"{d}: TPR={tpr:.3f}, FPR={fpr:.3f}")
        except Exception as e:
            print(f"[eval] {d} failed: {e}")

    print("\n=== Summary ===")
    print(f"{'Dataset':<20} {'TPR':>8} {'FPR':>8}")
    for d, t, f in results:
        print(f"{d:<20} {t:>8.3f} {f:>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["attack", "train", "eval", "all"],
                        default="all")
    parser.add_argument("--output-dir", default=None,
                        help="既存の output ディレクトリを再利用する場合に指定")
    args = parser.parse_args()

    output_dir = args.output_dir or create_run_dir()
    print(f"Output dir: {output_dir}")

    if args.phase in ("attack", "all"):
        run_attack(output_dir)
    if args.phase in ("train", "all"):
        run_train(output_dir)
    if args.phase in ("eval", "all"):
        run_eval(output_dir)


if __name__ == "__main__":
    main()
