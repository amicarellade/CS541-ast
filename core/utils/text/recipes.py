import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hparams as hp
from utils.fileutils import get_files


def ljspeech(path) :

    csv_file = get_files(path, extension='.csv')

    assert len(csv_file) == 1

    text_dict = {}

    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[-1]

    return text_dict

def nick(path) :

    df = pd.read_csv(os.path.join(path, hp.vctk_csv))

    text_dict = {}

    for _, row in df.iterrows():
        file_name = str(row["file_name"])
        text_dict[file_name[:file_name.index('.')]] = row["sentence"]

    return text_dict

    """
    csv_file = get_files(path, extension='.csv')

    assert len(csv_file) == 1

    text_dict = {}

    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split(',')
            text_dict[split[0]] = split[-2]

    return text_dict
    """
