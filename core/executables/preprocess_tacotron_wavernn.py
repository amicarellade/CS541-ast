import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loggingutils import *
import hparams as hp
from multiprocessing import Pool, cpu_count, set_start_method
from utils.fileutils import *
from utils.oputils import *
import pickle
import argparse
from utils.text.recipes import *


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', default=hp.wav_path, help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', default='.wav', help='file extension to search for in dataset folder')
args = parser.parse_args()

extension = args.extension
path = args.path

paths = Paths(hp.wav_path, hp.voc_model_id, hp.tts_model_id)

def convert_file(path) :
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW' :
        quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL' :
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path) :
    id = path.split(hp.f_delim)[-1][:-4]
    m, x = convert_file(path)
    np.save(f'{paths.mel}{id}.npy', m, allow_pickle=False)
    np.save(f'{paths.quant}{id}.npy', x, allow_pickle=False)
    return id, m.shape[-1]

if __name__ == "__main__":
    set_start_method("spawn")

    wav_files = get_files(path, extension)

    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

    if len(wav_files) == 0 :

        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')

    else :

        if not hp.ignore_tts :
        
            text_dict = nick(os.path.dirname(path))

            with open(f'{paths.data}text_dict.pkl', 'wb') as f:
                pickle.dump(text_dict, f)

        simple_table([('Sample Rate', hp.sample_rate),
                    ('Bit Depth', hp.bits),
                    ('Mu Law', hp.mu_law),
                    ('Hop Length', hp.hop_length),
                    ('CPU Count', cpu_count())])

        pool = Pool(processes=cpu_count())

        dataset = []

        for i, (id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
            dataset += [(id, length)]
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)

        with open(f'{paths.data}dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
