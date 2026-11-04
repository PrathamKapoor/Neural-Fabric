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
