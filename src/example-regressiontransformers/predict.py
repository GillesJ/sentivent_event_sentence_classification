import argparse
from copy import deepcopy
from itertools import product
import json
import logging
import os
from pathlib import Path
import random

import numpy as np
from pytorch_transformers.optimization import AdamW
from shutil import move
import torch
from torch import optim

from transformers import TransformerRegressor, TransformerTokenizer
from trainer import TransformerRegressionTrainer

logging.basicConfig(
    datefmt="%d-%b %H:%M:%S",
    format="%(asctime)s - [%(levelname)s]: %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("process.log"), logging.StreamHandler()],
)


class TransformerRegressionPredictor:
    """ Entry point to start training. Initializes and trains models in `predict()`. """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def set_seed():
        """ Set all seeds to 3 to make results reproducible (deterministic mode) """
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(3)
        random.seed(3)
        os.environ["PYTHONHASHSEED"] = str(3)

    def predict(self):
        """ Using the configuration file, initializes model, optimizer, scheduler and so on.
            Currently you can train on different configurations by passing a list to dropout and lr.
            All possible combinations will be tested.
            Will save the best model checkpoint for each configuration as well as its configuration file,
            and a loss graph. Saved to 'output_dir' in config.
        """
        for drop, lr in product(
            self.config["model"]["dropout"], self.config["optimizer"]["lr"]
        ):
            self.set_seed()
            opts = deepcopy(self.config)
            output_p = Path(opts["training"].pop("output_dir")).resolve()
            output_p.mkdir(exist_ok=True, parents=True)

            opts["optimizer"]["lr"] = lr
            opts["model"]["dropout"] = drop

            model = TransformerRegressor(opts["model"])
            tokenizer = TransformerTokenizer(opts["model"])

            logging.info(f"{opts['model']['name']} model and tokenizer loaded!")

            optimizer, optim_name = self.get_optim(opts["optimizer"], model)
            scheduler = self.get_scheduler(opts["scheduler"], optimizer)

            trainer = TransformerRegressionTrainer(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                **opts["training"]["files"],
                gpu_ids=opts["training"]["gpu_ids"],
                batch_size=opts["training"]["batch_size"],
                patience=opts["training"]["patience"],
            )

            best_model_f, fig = trainer.train(epochs=opts["training"]["epochs"])
            best_model_p = Path(best_model_f).resolve()
            loss, pearson = trainer.test()

            s = self.get_output_prefix(loss, pearson, opts)
            model_out = output_p.joinpath(s + "model.pth")
            config_out = output_p.joinpath(s + "config.json")

            # write config file based on actual values
            with config_out.open("w", encoding="utf-8") as fhout:
                json.dump(opts, fhout)

            # move output model to output_dir
            move(best_model_p, model_out)
            # Save plot
            fig.savefig(output_p.joinpath(s + "plot.png"))

    @staticmethod
    def get_output_prefix(loss, pearson, opts):
        """ Create output prefix for the output files based on current config and results """
        mopts = opts["model"]
        oopts = opts["optimizer"]
        sopts = opts["scheduler"]

        s = f"loss{loss:.2f}-pearson{pearson:.2f}-"
        s += f"{oopts['name']}-lr{oopts['lr']:.0E}-"
        s += f"{mopts['name']}-{mopts['weights']}-drop{mopts['dropout']:.2f}-"
        s += f"{sopts['name']}-" if "name" in sopts and sopts["name"] else ""

        return s

    @staticmethod
    def get_optim(optim_obj, model):
        """ Get the optimizer based on current config. Currently only AdamW and Adam are supported. """
        optim_copy = deepcopy(optim_obj)
        optim_name = optim_copy.pop("name")
        if optim_name.lower() == "adamw":
            return (
                AdamW([p for p in model.parameters() if p.requires_grad], **optim_copy),
                optim_name,
            )
        elif optim_name.lower() == "adam":
            return (
                optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], **optim_copy
                ),
                optim_name,
            )
        else:
            raise NotImplementedError(
                "This optimiser has not been implemented in this script."
            )

    @staticmethod
    def get_scheduler(sched_obj, optimizer):
        """ Get the scheduler based on current config. Currently only ReduceLROnPlateau is supported. """
        sched_copy = deepcopy(sched_obj)
        sched_name = sched_obj.pop("name")
        if sched_name:
            if sched_name.lower() == "reducelronplateau":
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", **sched_copy
                )
            else:
                raise NotImplementedError(
                    "This scheduler has not been implemented in this system yet."
                )
        else:
            return None


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train regression model.")
    arg_parser.add_argument(
        "config_f", help="Path to JSON file with configuration options."
    )

    cli_args = arg_parser.parse_args()
    with open(cli_args.config_f, "r") as config_fh:
        options = json.load(config_fh)

    predictor = TransformerRegressionPredictor(options)
    predictor.predict()
