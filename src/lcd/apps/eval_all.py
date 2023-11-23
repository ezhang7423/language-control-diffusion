import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from eztils import datestr
from eztils.torch import seed_everything
from IPython.display import display
from rich import print

from lcd import DATA_PATH
from lcd.utils.clevr.eval import DryEvalArgs, evaluate

MODEL_PATH = DATA_PATH / "models"


def eval_over_seeds(
    path: Path, path_key: str, model_file: str, args: DryEvalArgs, **kwargs
):
    results = []
    modes = ["train", "test", "error"]
    for seed in path.iterdir():
        print("Evaluating:", seed.name)

        setattr(args, path_key, str(seed / model_file))
        res = evaluate(dry_eval_args=args, eval_all=True, **kwargs)
        results.append([res[f"eval/{mode}-sr"] for mode in modes])

    raw_results = np.array(results).T
    return raw_results.mean(axis=1), raw_results.std(axis=1), raw_results


def fmt(mean, std):
    # round the numbers to 2 decimals
    # return f'{mean[0]} ± {std[0]}'
    mean *= 100
    std *= 100
    return f"{mean:.1f} ± {std:.1f}"


def main(
    llp: bool = False,
    transformer: bool = False,
    diffuser: bool = False,
    lcd: bool = False,
    hierarchical: bool = True,
):
    seed_everything(0)
    df = pd.DataFrame(columns=["model", "train", "test", "error"])
    raw = {}

    def add_to_df(model, mean, std, results):
        new_df = df.append(
            {
                "model": model,
                "train": fmt(mean[0], std[0]),
                "test": fmt(mean[1], std[1]),
                "error": fmt(mean[2], std[2]),
            },
            ignore_index=True,
        )

        raw[model] = results.tolist()
        return new_df

    ###################
    # llp
    ###################
    if llp:
        ret = evaluate(
            DryEvalArgs(
                dataset_path=DATA_PATH / "ball_llp_eval.pt",
                num_sequences=100,                        
            ),
            num_processes=1
        )
        print("LLP results:", ret)

    ###################
    # e2e transformer
    ###################

    if transformer:
        print("Evaluating transformer...")

        path = MODEL_PATH / "transformer"
        args = DryEvalArgs(transformer=True, only_hlp=True, num_sequences=100)
        mean, std, results = eval_over_seeds(
            path, "high_model_path", "model_9.pt", args, num_processes=1
        )
        df = add_to_df("transformer", mean, std, results)

    ###################
    # diffuser
    ###################

    if diffuser:
        print("Evaluating diffuser...")

        path = MODEL_PATH / "diffuser"
        args = DryEvalArgs(
            num_sequences=2,
            only_hlp=True,
            transformer=False,
        )

        mean, std, results = eval_over_seeds(
            path,
            "high_model_path",
            model_file="model_100000.pt",
            args=args,
            num_processes=50,
        )

        df = add_to_df("diffuser", mean, std, results)

    ###################
    # lcd
    ###################

    if lcd:
        print("Evaluating lcd...")

        path = MODEL_PATH / "lcd"
        args = DryEvalArgs(
            num_sequences=2,
            only_hlp=False,
            transformer=False,
        )

        mean, std, results = eval_over_seeds(
            path,
            "high_model_path",
            model_file="model_200000.pt",
            args=args,
            num_processes=50,
        )

        df = add_to_df("lcd", mean, std, results)

    ###################
    # hierarchical transformer
    ###################

    if hierarchical:
        print("Evaluating hierarchical transformer...")

        path = MODEL_PATH / "hierarchical-transformer/"
        args = DryEvalArgs(
            num_sequences=100,
            only_hlp=False,
            transformer=True,
            dataset_path=None,
        )

        mean, std, results = eval_over_seeds(
            path,
            "high_model_path",
            model_file="model_9.pt",
            args=args,
            num_processes=1,
        )

        df = add_to_df("hierarchical", mean, std, results)
        
    ###################
    # save results
    ###################

    with io.StringIO() as buffer:
        df.to_csv(buffer, sep=" ", index=False)
        print(buffer.getvalue())

    print("Copying to clipboard")
    df.to_clipboard()

    # save the dataframe
    print("Saving to csv: ", os.getcwd() + f"/results_{datestr()}.csv")
    df.to_csv(f"results_{datestr()}.csv")

    # save the raw results
    print(
        "Saving raw results to json: ", os.getcwd() + f"/raw_results_{datestr()}.json"
    )
    with open(f"raw_results_{datestr()}.json", "w") as f:
        json.dump(raw, f)


if __name__ == "__main__":
    typer.run(main)
