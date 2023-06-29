import argparse
import pickle
import os
import torch

from .data import VALLEDatset, create_train_val_dataloader
from .train import load_engines


def main():
    parser = argparse.ArgumentParser("Save trained model to path.")
    parser.add_argument("path")
    args = parser.parse_args()

    engine = load_engines()
    ckpt = torch.load('/data/vall-e/ckpts_style/korean/nar/model/style/mp_rank_00_model_states_exported.pt')    

    engine["model"].load_state_dict(ckpt)


if __name__ == "__main__":
    main()
