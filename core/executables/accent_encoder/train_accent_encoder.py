import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import hparams as hp

from sklearn.preprocessing import LabelEncoder




def preprocess_df(df: pd.DataFrame):

    # Encodings for accents, OHE vector will be supplied in dataset
    encoder = LabelEncoder()
    accents = encoder.fit(df["accents"])
    df["enc_accents"] = accents

    return df


def train_classifier(model: nn.Module):


    def train():
        pass

    def validate():
        pass



if __name__ == "__main__":

    df = pd.read_csv(os.path.join(hp.data_path, hp.vctk_csv))

    df = preprocess_df(df)


    pass