import argparse
import json
import sys
from .checkpoint import load
from .workflow import train, evaluate, predict


def main():
    parser = argparse.ArgumentParser(prog="neural_fabric")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train-demo")
    t.add_argument("--output", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=20)
    ct = commands.add_parser("train-csv")
    ct.add_argument("--csv", required=True)
    ct.add_argument("--target", required=True)
    ct.add_argument("--output", required=True)
    ct.add_argument("--seed", type=int, default=42)
    ct.add_argument("--epochs", type=int, default=20)
    for name in ("evaluate-csv", "predict-csv"):
        cp = commands.add_parser(name)
        cp.add_argument("--model", required=True)
        cp.add_argument("--csv", required=True)
        cp.add_argument("--mode", choices=("soft", "top1"), default="top1")
    for name in ("predict", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--mode", choices=("soft", "top1", "topk"), default="top1")
        p.add_argument("--k", type=int, default=1)
        if name == "predict":
            p.add_argument("--input", required=True)
        else:
            p.add_argument("--output", help="Write result JSON to a new file; never overwrites")
    args = parser.parse_args()
    try:
        if args.command == "train-demo":
            result = train(args.output, args.seed, args.epochs)
        elif args.command in ("train-csv", "evaluate-csv", "predict-csv"):
            from .csv_workflow import train_csv, evaluate_csv, predict_csv
            if args.command == "train-csv":
                result = train_csv(args.csv, args.target, args.output, args.seed, args.epochs)
            elif args.command == "evaluate-csv":
                result = evaluate_csv(args.model, args.csv, args.mode)
            else:
                result = predict_csv(args.model, args.csv, args.mode)
        else:
            model, checkpoint = load(args.model)
            if args.command == "predict":
                with open(args.input, encoding="utf-8") as handle:
                    value = json.load(handle)
                result = predict(model, value["input"], args.mode, args.k)
            else:
                result = evaluate(model, checkpoint["metadata"]["seed"], args.mode, args.k)
                if args.output:
                    with open(args.output, "x", encoding="utf-8") as handle:
                        json.dump(result, handle, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
