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
    model = engine["model"].module.cpu()
    
    print('loading dataloader...')
    # load dataloader
    if os.path.isfile('/data/vall-e/vall_e/dataloaders/train_dl.pkl'):  
        with open('/data/vall-e/vall_e/dataloaders/train_dl.pkl', 'rb') as f:
            train_dl = pickle.load(f)
    else:
        train_dl, *_ = create_train_val_dataloader()
    # train_dl, *_ = create_train_val_dataloader()
    
    print('loaded dataloader...')
    
    assert isinstance(train_dl.dataset, VALLEDatset)
    model.phone_symmap = train_dl.dataset.phone_symmap
    model.spkr_symmap = train_dl.dataset.spkr_symmap
    
    print('saving model...')
    torch.save(model, args.path)
    print(args.path, "saved.")


if __name__ == "__main__":
    main()
