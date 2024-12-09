import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accent_encoder.accent_style_transfer import *
from utils.extras.acc_encoder_preprocess_utils import *
from executables.wavernn.gen_wavernn import *
from executables.tacotron.gen_tacotron import get_spk_embed
from utils.text import text_to_sequence
from utils.text.symbols import symbols
from models.tacotron import Tacotron

def wavs_only_nst_pregen_spk_embeds(content_ex_path: str, style_ex_path: str, spk_embed_path: str, perform_nst: bool = True):
    """
    Neural Style Transfer on pre-recorded audio files (supported formats: .wav)\n
    Runs Accent Neural Style Transfer and WaveRNN
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if perform_nst:

        content_spec = extract_mel_spectrogram(content_ex_path)
        style_spec = extract_mel_spectrogram(style_ex_path)

        length_content = content_spec.shape[1]
        length_style = style_spec.shape[1]

        content_spec = resize_spectrogram(content_spec)
        style_spec = resize_spectrogram(style_spec)

        model = VGG19Net(num_classes = 11)
        weights = torch.load(os.path.join(hp.vgg_path, "vgg_acc.pth"), map_location=device)
        model.vgg.load_state_dict(weights)

        output = neural_style_transfer(model, content_spec, style_spec, device, style_weight=1000000)

        save_mels = True

        if save_mels:
            print("Saving Spectrograms")
            save_spectrogram_dual(content_spec[:, :length_content], os.path.join(hp.output, "content.png"), os.path.join(hp.output, "content.npy"), True)
            save_spectrogram_dual(style_spec[:, :length_style], os.path.join(hp.output, "style.png"), os.path.join(hp.output, "style.npy"), True)
            save_spectrogram_dual(output[:, :length_content], os.path.join(hp.output, "new.png"), os.path.join(hp.output, "new.npy"), True)
    else:
        output = np.load(os.path.join(hp.output, "new.npy"))


    output = output[:, :length_content]
    output = normalize(output)

    # WaveRNN
    model = SCWaveRNN(rnn_dims=hp.voc_rnn_dims,
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

    spk_embed = np.load(spk_embed_path)

    wavernn_weights = torch.load(hp.best_wavernn, map_location=device)
    model = model.to(device)

    model.load_state_dict(wavernn_weights)

    model.generate(torch.Tensor(output).unsqueeze(0), torch.Tensor(spk_embed).unsqueeze(0), os.path.join(hp.output, "final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)

    print("Done")


def full_nst_pregen_spk_embeds(sentence: str, content_spk_embed_path: str, style_spk_embed_path: str):
    """
    Full Neural Style Transfer Pipeline using pre-generated speaker embeddings\n
    Runs Tacotron, the Accent Neural Style Transfer, and WaveRNN
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    content_spk_embed = np.load(content_spk_embed_path)
    style_spk_embed = np.load(style_spk_embed_path)

    # Tacotron
    print("-----Tacotron-----\n")
    seq = text_to_sequence(sentence.strip(), hp.tts_cleaner_names)

    tacotron = Tacotron(embed_dims=hp.tts_embed_dims,
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
                         dropout=hp.tts_dropout).to(device)
    
    tacotron.load_state_dict(torch.load(paths.tts_latest_weights, map_location=device))
    tacotron.decoder.r = tacotron.r.item()

    _, content_mel_spec, cs_attention = tacotron.generate(seq, s_e=torch.Tensor(content_spk_embed).unsqueeze(0))
    _, style_mel_spec, ss_attention = tacotron.generate(seq, s_e=torch.Tensor(style_spk_embed).unsqueeze(0))

    content_mel_spec = (content_mel_spec + 4) / 8
    style_mel_spec = (style_mel_spec + 4) / 8

    save_attention(cs_attention, os.path.join(hp.output, "cs_attn"))
    save_attention(ss_attention, os.path.join(hp.output, "ss_attn"))

    # Neural Style Transfer
    print("-----Neural Style Transfer-----\n")

    length_content = content_mel_spec.shape[1]
    length_style = style_mel_spec.shape[1]

    content_mel_spec = resize_spectrogram(content_mel_spec)
    style_mel_spec = resize_spectrogram(style_mel_spec)

    model = VGG19Net(num_classes = 11)
    weights = torch.load(os.path.join(hp.vgg_path, "vgg_acc.pth"), map_location=device)
    model.vgg.load_state_dict(weights)

    output = neural_style_transfer(model, content_mel_spec, style_mel_spec, device)

    save_mels = True

    if save_mels:
        print("Saving Spectrograms")
        save_spectrogram_dual(content_mel_spec[:, :length_content], os.path.join(hp.output, "content.png"), os.path.join(hp.output, "content.npy"), True)
        save_spectrogram_dual(style_mel_spec[:, :length_style], os.path.join(hp.output, "style.png"), os.path.join(hp.output, "style.npy"), True)
        save_spectrogram_dual(output[:, :length_content], os.path.join(hp.output, "new.png"), os.path.join(hp.output, "new.npy"), True)

    # WaveRNN
    print("-----WaveRNN-----\n")
    wavernn = SCWaveRNN(rnn_dims=hp.voc_rnn_dims,
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
                    mode=hp.voc_mode).to(device)

    wavernn.load_state_dict(torch.load(hp.best_wavernn, map_location=device))

    wavernn.generate(torch.Tensor(content_mel_spec).unsqueeze(0), torch.Tensor(content_spk_embed).unsqueeze(0), os.path.join(hp.output, "content_final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)
    wavernn.generate(torch.Tensor(style_mel_spec).unsqueeze(0), torch.Tensor(style_spk_embed).unsqueeze(0), os.path.join(hp.output, "style_final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)
    wavernn.generate(torch.Tensor(output).unsqueeze(0), torch.Tensor(content_spk_embed).unsqueeze(0), os.path.join(hp.output, "new_final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)


def full_nst_gen_spk_embeds(sentence: str, content_spk_wav_path: str, style_spk_wav_path: str, enc_model_path: str = f"{hp.spk_encoder_model_path}{hp.f_delim}pretrained.pt"):
    """
    Full Neural Style Transfer Pipeline using unseen speakers\n
    Runs the Speaker Encoder, Tacotron, the Accent Neural Style Transfer, and WaveRNN
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    content_spk_embed, _ = get_spk_embed(content_spk_wav_path, enc_model_path)
    style_spk_embed, _  = get_spk_embed(style_spk_wav_path, enc_model_path)

    # Tacotron
    print("-----Tacotron-----\n")
    seq = text_to_sequence(sentence.strip(), hp.tts_cleaner_names)

    tacotron = Tacotron(embed_dims=hp.tts_embed_dims,
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
                         dropout=hp.tts_dropout).to(device)
    
    tacotron.load_state_dict(torch.load(paths.tts_latest_weights, map_location=device))
    tacotron.decoder.r = tacotron.r.item()

    _, content_mel_spec, cs_attention = tacotron.generate(seq, s_e=torch.Tensor(content_spk_embed).unsqueeze(0))
    _, style_mel_spec, ss_attention = tacotron.generate(seq, s_e=torch.Tensor(style_spk_embed).unsqueeze(0))

    content_mel_spec = (content_mel_spec + 4) / 8
    style_mel_spec = (style_mel_spec + 4) / 8

    save_attention(cs_attention, os.path.join(hp.output, "cs_attn"))
    save_attention(ss_attention, os.path.join(hp.output, "ss_attn"))

    # Neural Style Transfer
    print("-----Neural Style Transfer-----\n")

    length_content = content_mel_spec.shape[1]
    length_style = style_mel_spec.shape[1]

    content_mel_spec = resize_spectrogram(content_mel_spec)
    style_mel_spec = resize_spectrogram(style_mel_spec)

    model = VGG19Net(num_classes = 11)
    weights = torch.load(os.path.join(hp.vgg_path, "vgg_acc.pth"), map_location=device)
    model.vgg.load_state_dict(weights)

    output = neural_style_transfer(model, content_mel_spec, style_mel_spec, device)

    save_mels = True

    if save_mels:
        print("Saving Spectrograms")
        save_spectrogram_dual(content_mel_spec[:, :length_content], os.path.join(hp.output, "content.png"), os.path.join(hp.output, "content.npy"), True)
        save_spectrogram_dual(style_mel_spec[:, :length_style], os.path.join(hp.output, "style.png"), os.path.join(hp.output, "style.npy"), True)
        save_spectrogram_dual(output[:, :length_content], os.path.join(hp.output, "new.png"), os.path.join(hp.output, "new.npy"), True)

    # WaveRNN
    print("-----WaveRNN-----\n")
    wavernn = SCWaveRNN(rnn_dims=hp.voc_rnn_dims,
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
                    mode=hp.voc_mode).to(device)

    wavernn.load_state_dict(torch.load(hp.best_wavernn, map_location=device))

    wavernn.generate(torch.Tensor(content_mel_spec).unsqueeze(0), torch.Tensor(content_spk_embed).unsqueeze(0), os.path.join(hp.output, "content_final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)
    wavernn.generate(torch.Tensor(style_mel_spec).unsqueeze(0), torch.Tensor(style_spk_embed).unsqueeze(0), os.path.join(hp.output, "style_final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)
    wavernn.generate(torch.Tensor(output).unsqueeze(0), torch.Tensor(content_spk_embed).unsqueeze(0), os.path.join(hp.output, "new_final.wav"), True, hp.voc_target, hp.voc_overlap, hp.mu_law)


if __name__ == "__main__":

    mode = 1

    if mode == 0:
        wavs_only_nst_pregen_spk_embeds(os.path.join(hp.output, "content_preprocess.wav"), 
                                        os.path.join(hp.output, "style_preprocess.wav"),
                                        f"{hp.wav_path}{hp.f_delim}vctk_mol{hp.f_delim}spk_embeds{hp.f_delim}p360_006.npy")
    elif mode == 1:
        full_nst_pregen_spk_embeds("When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
                                f"{hp.wav_path}{hp.f_delim}vctk_mol{hp.f_delim}spk_embeds{hp.f_delim}p360_006.npy",
                                f"{hp.wav_path}{hp.f_delim}vctk_mol{hp.f_delim}spk_embeds{hp.f_delim}p236_006.npy")
    elif mode == 2:
        full_nst_gen_spk_embeds("When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
                                os.path.join(hp.output, "content_preprocess.wav"), 
                                os.path.join(hp.output, "style_preprocess.wav"))

    pass

