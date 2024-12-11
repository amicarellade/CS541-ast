import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.loggingutils import *
from datasets import get_tts_dataset
import hparams as hp
from utils.text.symbols import symbols
from utils.fileutils import Paths
from models.tacotron import Tacotron
from train_tacotron import create_gta_features

if __name__ == "__main__" :

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     n_mels=hp.num_mels,
                     fft_bins=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout).cuda()

    paths = Paths(hp.wav_path, hp.voc_model_id, hp.tts_model_id)

    model.restore(paths.tts_latest_weights)

    print('Creating Ground Truth Aligned Dataset...\n')

    train_set, attn_example = get_tts_dataset(paths.data, 8, model.get_r())
    create_gta_features(model, train_set, paths.gta)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')