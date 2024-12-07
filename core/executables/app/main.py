import sys
import os
import librosa
import soundfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accent_encoder.accent_style_transfer import *
from utils.extras.acc_encoder_preprocess_utils import *



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output = f"{hp.output}\\test.wav"

    content_ex_path = os.path.join(hp.data_path, "p360\\p360_006.wav")
    style_ex_path = os.path.join(hp.data_path, "p256\\p256_006.wav")
    style_ex_path2 = os.path.join(hp.data_path, "p376\\p376_006.wav")

    content_spec = resize_spectrogram(extract_mel_spectrogram(content_ex_path))
    style_spec = resize_spectrogram(extract_mel_spectrogram(style_ex_path))

    model = VGG19Net(num_classes = 11)
    weights = torch.load(os.path.join(hp.vgg_path, "vgg_acc.pth"), map_location=device)
    model.vgg.load_state_dict(weights)

    output = neural_style_transfer(model, content_spec, style_spec, device)

    save_spectrogram(content_spec, os.path.join(hp.output, "content.png"), os.path.join(hp.output, "content.npy"), True)
    save_spectrogram(style_spec, os.path.join(hp.output, "style.png"), os.path.join(hp.output, "style.npy"), True)
    save_spectrogram(output, os.path.join(hp.output, "new.png"), os.path.join(hp.output, "new.npy"), True)

    print("Done")

