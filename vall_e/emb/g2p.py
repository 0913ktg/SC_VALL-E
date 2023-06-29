import argparse
import random
import string
from functools import cache
from pathlib import Path
import re


import torch
# from g2pk import G2p
from vall_e.emb.G2P.KoG2Padvanced import KoG2Padvanced as G2p
from vall_e.emb.G2P.KoG2P import KoG2P
from tqdm import tqdm


@cache
def _get_model():
    return G2p()


@cache
def _get_graphs(path):
    with open(path, "r") as f:
        graphs = f.read()
    
    return graphs


def encode(graphs: str) -> list[str]:
    # g2p = _get_model()
    # phones = g2p(graphs)
    pattern = '[^ㄱ-ㅎㅏ-ㅣ가-힣]+'
    graphs = re.sub(pattern, ' ', graphs)
    outputText = G2p(graphs)
    outputText = outputText.split(' ')
    new_list = []
    for output in outputText:
        new_list.append(str(KoG2P(output)))
    phones = ' _ '.join(new_list)
    # ignored = {" ", *string.punctuation}
    # return ["_" if p in ignored else p for p in phones]
    return phones


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--suffix", type=str, default=".normalized.txt")
    args = parser.parse_args()

    paths = list(args.folder.rglob(f"*{args.suffix}"))
    random.shuffle(paths)

    for path in tqdm(paths):
        phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
        if phone_path.exists():
            continue
        graphs = _get_graphs(path)
        try:
            phones = encode(graphs)
            with open(phone_path, "w") as f:
                f.write(phones)
        except KeyError:
            print(path)



if __name__ == "__main__":
    main()