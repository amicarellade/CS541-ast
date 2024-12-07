import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import hparams as hp

# Paths
BASE_DIR = hp.base_vctk  # Base directory of the VCTK corpus
WAV_DIR = os.path.join(BASE_DIR, 'wavs')  # Path to wav files directory
CSV_FILE = os.path.join(BASE_DIR, 'speaker-info.csv')  # Path to speaker info CSV
OUTPUT_DIR =  hp.output # os.path.join(BASE_DIR, 'mel-spec')  # Directory to save spectrograms

# Create output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters for audio processing (aligned with the main repo)
SAMPLE_RATE = 22050  # Updated sample rate
N_MELS = 80          # Number of mel bands (aligned with main repo)
HOP_LENGTH = 275     # Hop length for FFT (aligned with Tacotron 2 paper)
WIN_LENGTH = 1100    # Window length for FFT
FMIN = 40            # Minimum frequency for mel filterbank
FMAX = None          # Maximum frequency (set to None for default)
REF_LEVEL_DB = 20    # Reference level for dB conversion
MIN_LEVEL_DB = -100  # Minimum level for dB normalization

# Output spectrogram size for resizing
TARGET_SIZE = hp.default_mel_dims  # (Height x Width), based on mel bands and average speech length


def extract_mel_spectrogram(file_path):
    """Extract Mel spectrogram from a WAV file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=hp.n_fft,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to dB
        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def resize_spectrogram(spectrogram, target_size=(80, 800)):
    """Resize a spectrogram to the target size (padding or truncating as needed)."""
    h, w = spectrogram.shape  # Original height and width
    target_h, target_w = target_size  # Target height and width

    # Handle width (time axis)
    if w > target_w:  # Truncate if too wide
        spectrogram = spectrogram[:, :target_w]
    elif w < target_w:  # Pad if too narrow
        pad_width = target_w - w
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    # Handle height (frequency axis)
    if h > target_h:  # Truncate if too tall
        spectrogram = spectrogram[:target_h, :]  # Crop height
    elif h < target_h:  # Pad if too short
        pad_height = target_h - h
        spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant')

    return spectrogram


def save_spectrogram(spectrogram, img_path, npy_path, save_pngs: bool = False):
    """Save spectrogram as both .npy file and .png image."""
    np.save(npy_path, spectrogram)  # Save as NumPy array

    # Save as an image
    if save_pngs:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spectrogram,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis='time',
            y_axis='mel',
            fmin=FMIN,
            fmax=FMAX,
        )
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(img_path)
        plt.close('all')


def preprocess_data(save_pngs: bool = False):
    # Load speaker info from CSV file
    metadata = pd.read_csv(CSV_FILE)
    metadata = metadata.rename(columns=lambda x: x.strip())  # Remove extra spaces in column names

    # Process each speaker
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        speaker_id = f"p{row['ID']}"  # Use 'ID' column for speaker directory name (e.g., p225)
        speaker_dir = os.path.join(WAV_DIR, speaker_id)
        output_speaker_dir = os.path.join(OUTPUT_DIR, speaker_id)

        # Skip if speaker directory is missing
        if not os.path.exists(speaker_dir):
            print(f"Missing directory for speaker {speaker_id}")
            continue

        # Create output directory for the speaker
        os.makedirs(output_speaker_dir, exist_ok=True)

        # Process all WAV files for the speaker
        for wav_file in os.listdir(speaker_dir):
            if wav_file.endswith('.wav'):
                file_path = os.path.join(speaker_dir, wav_file)
                mel_spectrogram = extract_mel_spectrogram(file_path)

                if mel_spectrogram is not None:
                    # Resize spectrogram to fixed dimensions
                    resized_spectrogram = resize_spectrogram(mel_spectrogram)

                    # Define paths for saving spectrogram
                    base_name = os.path.splitext(wav_file)[0]
                    npy_path = os.path.join(output_speaker_dir, f"{base_name}.npy")

                    img_path = os.path.join(output_speaker_dir, f"{base_name}.png")

                    # Save spectrogram as both .npy and .png
                    save_spectrogram(resized_spectrogram, img_path, npy_path, save_pngs)

    print(f"Preprocessing complete. Spectrograms saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess_data()