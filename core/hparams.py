import os

abs_path = os.path.dirname(os.path.abspath(__file__))

# CONFIG -----------------------------------------------------------------------------------------------------------#

# '/' if Linux, '\\' if Windows
f_delim = '\\'

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
base_vctk = 'D:\\Users\\ibarn\\Documents\\Dataset Repository\\audio\\VCTK-Corpus'

wav_path = 'D:\\Users\\ibarn\\Documents\\Dataset Repository\\audio\\VCTK-Corpus\\wavs'
data_path = 'D:\\Users\\ibarn\\Documents\\Dataset Repository\\audio\\VCTK-Corpus\\wavs'

# CSV names
vctk_csv = "metadata.csv"

# model ids are separate - that way you can use a new tts with an old wavernn and vice versa
# NB: expect undefined behaviour if models were trained on different DSP settings
voc_model_id = 'vctk_mol'
tts_model_id = 'vctk_lsa_smooth_attention'

# set this to True if you are only interested in WaveRNN
ignore_tts = False


# Global model weights save path
models_save_path = os.path.join(abs_path, "model_weights")

output = "D:\\Users\\ibarn\\Documents\\VSCode Repos\\specific\\WPI Projects\\python\\CS 541\\FinalProject\\CS541-ast\\out"


# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
win_length = 1100                   # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# SPEAKER ENCODER --------------------------------------------------------------------------------------------------#

spk_encoder_model_path = f"{models_save_path}{f_delim}spk_encoder{f_delim}"

se_data_path = "D:\\Users\\ibarn\\Documents\\Dataset Repository\\audio\\se_dataset"
default_run_id = "pre_accent_encoded"

librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]


""" ---- Data Parameters ---- """

# For logging
spk_encoder_params = ["mel_window_length", 
                      "mel_window_step", 
                      "mel_n_channels",
                      "sampling_rate",
                      "partials_n_frames",
                      "inference_n_frames",
                      "vad_window_length",
                      "vad_moving_average_width",
                      "vad_max_silence_length",
                      "audio_norm_target_dBFS"]

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30

""" ---- Model Parameters ---- """

## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3


## Training parameters
learning_rate_init = 1e-4
speakers_per_batch = 256 # 64
utterances_per_speaker = 10

# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 64
voc_lr = 1e-4
voc_checkpoint_every = 25_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 1_000_000         # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True         # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_r = 1                           # model predicts r frames per output step
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']

# Training


tts_schedule = [(7,  1e-3,  10_000,  32),   # progressive training schedule r=7, bs=32
                (5,  1e-4, 100_000,  32),   # (r, lr, step, batch_size) Note: r = number of spectrogram frames predicted per decoder iteration r=5, bs=32
                (2,  1e-4, 180_000,  16),   # r=2, bs=16
                (2,  1e-4, 350_000,  16),   # r=2, bs=16
                (2,  1e-4, 600_000,  16),   # r=2, bs=16
                (1,  1e-4, 850_000,  8),    # r=1, bs=8
                (1,  1e-4, 1000_000,  8)]   # r=1, bs=8

tts_max_mel_len = 2500              # if you have a couple of extremely long spectrograms you might want to use this default: (1250)
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_checkpoint_every = 2_000        # checkpoints the model every X steps
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes

# ACCENT ENCODER ---------------------------------------------------------------------------------------------------#

vgg_path = os.path.join(abs_path, f"model_weights{f_delim}acc_encoder{f_delim}vgg")

default_mel_dims = (80, 800)

# ------------------------------------------------------------------------------------------------------------------#

