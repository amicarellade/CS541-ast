import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from models.sc_wavernn import SCWaveRNN
import hparams as hp
from utils.fileutils import load_wav, Paths
from utils.text.symbols import symbols
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.loggingutils import save_attention, simple_table
import executables.speaker_encoder.generate_embeddings as encoder
from pathlib import Path

def get_spk_embed(load_path, enc_model_fpath) :
    
    file_name = load_path.split('/')[-1]
    wav = load_wav(load_path)
    encoder.load_model(enc_model_fpath)
    preprocessed_wav = encoder.preprocess_wav(load_path)
    embed = encoder.embed_utterance(preprocessed_wav)
    spk_embd = torch.tensor(embed).unsqueeze(0)

    return spk_embd, file_name



if __name__ == "__main__" :

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--file', '-f', type=str, help='[string/path] for testing a wav outside dataset for reference', default=os.path.join(hp.wav_path, "p360\\p360_006.wav"))
    parser.add_argument('--weights_path', '-w', type=str, help='[string/path] Load in different Tacotron Weights')
    parser.add_argument('--save_attention', '-a', dest='save_attn', action='store_true', help='Save Attention Plots')
    parser.add_argument("-e", "--enc_model_fpath", type=Path, default=f"{hp.spk_encoder_model_path}{hp.f_delim}pretrained.pt",help="Path to a saved encoder")
    parser.add_argument('--weights_voc', '-wv', type=str, help='[string/path] checkpoint file to load weights from')
    parser.add_argument("--output", "-out", type=Path, help="output path", default=hp.output)
    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(target=hp.voc_target)
    parser.set_defaults(overlap=hp.voc_overlap)
    parser.set_defaults(input_text=None)
    parser.set_defaults(weights_path=None)
    parser.set_defaults(save_attention=True)
    args = parser.parse_args()

    batched = args.batched
    target = args.target
    overlap = args.overlap
    input_text = args.input_text
    weights_path = args.weights_path
    save_attn = args.save_attention
    enc_path = args.enc_model_fpath
    files = args.file
    out = args.output

    paths = Paths(hp.wav_path, hp.voc_model_id, hp.tts_model_id)

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = SCWaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).cuda()


    restore_path_voc = hp.best_wavernn
    voc_model.restore(restore_path_voc)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
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

    tts_restore_path = weights_path if weights_path else paths.tts_latest_weights
    tts_model.restore(tts_restore_path)

    if input_text :
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else :
        with open(f'{hp.abs_path}{hp.f_delim}_misc{hp.f_delim}sentences.txt') as f :
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    simple_table([('WaveRNN', str(voc_k) + 'k'),
                  ('Tacotron', str(tts_k) + 'k'),
                  ('r', tts_model.r.item()),
                  ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    for i, x in enumerate(inputs, 1) :
  
        spk_embds, file_name = get_spk_embed(files, enc_path)

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x, spk_embds)

        if input_text :
           # save_path = f'{paths.tts_output}__input_{input_text[:10]}_{tts_k}k.wav'
            save_path = f'{out}{i}_output_batched{str(batched)}_{tts_k}k.wav'
        else :
            save_path = f'{out}{i}_output_batched{str(batched)}_{tts_k}k.wav'

        if save_attn : save_attention(attention, save_path)

        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8

        voc_model.generate(m, spk_embds, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

    print('\n\nDone.\n')
