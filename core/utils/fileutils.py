import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import librosa
import os
import hparams as hp
import numpy as np


class Paths :
    def __init__(self, data_path, voc_id, tts_id) :
        # Data Paths
        self.data = f'{data_path}{hp.f_delim}{voc_id}{hp.f_delim}'
        self.quant = f'{self.data}quant{hp.f_delim}'
        self.mel = f'{self.data}mel{hp.f_delim}'
        self.gta = f'{self.data}gta{hp.f_delim}'
        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = f'{hp.models_save_path}{hp.f_delim}wavernn{hp.f_delim}checkpoints{hp.f_delim}{voc_id}.wavernn{hp.f_delim}'
        self.voc_latest_weights = f'{self.voc_checkpoints}latest_weights.pyt'
        self.voc_output = f'{hp.models_save_path}{hp.f_delim}wavernn{hp.f_delim}model_outputs{hp.f_delim}{voc_id}.wavernn{hp.f_delim}'
        self.voc_step = f'{self.voc_checkpoints}{hp.f_delim}step.npy'
        self.voc_log = f'{self.voc_checkpoints}log.txt'
        # Tactron/TTS Paths
        self.tts_checkpoints = f'{hp.models_save_path}{hp.f_delim}tacotron{hp.f_delim}checkpoints{hp.f_delim}{tts_id}.tacotron{hp.f_delim}'
        self.tts_latest_weights = f'{self.tts_checkpoints}latest_weights.pyt'
        self.tts_output = f'{hp.models_save_path}{hp.f_delim}tacotron{hp.f_delim}model_outputs{hp.f_delim}{tts_id}.tts{hp.f_delim}'
        self.tts_step = f'{self.tts_checkpoints}{hp.f_delim}step.npy'
        self.tts_log = f'{self.tts_checkpoints}log.txt'
        self.tts_attention = f'{self.tts_checkpoints}{hp.f_delim}attention{hp.f_delim}'
        self.tts_mel_plot = f'{self.tts_checkpoints}{hp.f_delim}mel_plots{hp.f_delim}'
        self.create_paths()

    def create_paths(self) :
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)
        os.makedirs(self.tts_checkpoints, exist_ok=True)
        os.makedirs(self.tts_output, exist_ok=True)
        os.makedirs(self.tts_attention, exist_ok=True)
        os.makedirs(self.tts_mel_plot, exist_ok=True)


def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames
    
def load_wav(path) :
    return librosa.load(path, sr=hp.sample_rate)[0]


def save_wav(x, path) :
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)