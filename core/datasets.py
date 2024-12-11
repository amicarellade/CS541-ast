import pickle
import random
import torch
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.loggingutils import *
from utils.oputils import *
from utils.extras.speaker_encoder_dataobjects import *
from functools import partial
import hparams as hp
from utils.text import text_to_sequence
from pathlib import Path


###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class VocoderDataset(Dataset) :
    def __init__(self, ids, path, train_gta=False) :
        self.metadata = ids
        self.mel_path = f'{path}gta{hp.f_delim}' if train_gta else f'{path}mel{hp.f_delim}'
        self.quant_path = f'{path}quant{hp.f_delim}'
        self.spk_embd_path= f'{path}spk_embeds{hp.f_delim}'


    def __getitem__(self, index) :
        id = self.metadata[index]
        m = np.load(f'{self.mel_path}{id}.npy')
        x = np.load(f'{self.quant_path}{id}.npy')
        s_e=np.load(f'{self.spk_embd_path}{id}.npy')
        return m, x, s_e

    def __len__(self) :
        return len(self.metadata)


def get_vocoder_datasets(path, batch_size, train_gta) :

    with open(f'{path}dataset.pkl', 'rb') as f :
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(train_ids, path, train_gta)
    test_dataset = VocoderDataset(test_ids, path, train_gta)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]
    
    spk_embd = [x[2] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)
    spk_embd = np.stack(spk_embd).astype(np.float32)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()
    spk_embd = torch.tensor(spk_embd)

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL' :
        y = label_2_float(y.float(), bits)

    return x, y, mels, spk_embd


###################################################################################
# Tacotron/TTS Dataset ############################################################
###################################################################################


def get_tts_dataset(path, batch_size, r) :

    with open(f'{path}dataset.pkl', 'rb') as f :
        dataset = pickle.load(f)

    dataset_ids = []
    mel_lengths = []

    for (id, len) in dataset :
        if len <= hp.tts_max_mel_len :
            dataset_ids += [id]
            mel_lengths += [len]

    with open(f'{path}text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f)

    train_dataset = TTSDataset(path, dataset_ids, text_dict)

    sampler = None

    if hp.tts_bin_lengths :
        sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)

    collate_fn = partial(collate_tts, r=r)
    train_set = DataLoader(train_dataset,
                           collate_fn=collate_fn,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=1,
                           pin_memory=True)

    longest = mel_lengths.index(max(mel_lengths))
    attn_example = dataset_ids[longest]

    # print(attn_example)

    return train_set, attn_example


class TTSDataset(Dataset):
    def __init__(self, path, dataset_ids, text_dict) :
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        id = self.metadata[index]
        x = text_to_sequence(self.text_dict[id], hp.tts_cleaner_names)
        mel = np.load(f'{self.path}mel{hp.f_delim}{id}.npy')
        spk_embed = np.load(f'{self.path}spk_embeds{hp.f_delim}{id}.npy')
        mel_len = mel.shape[-1]
        return x, mel, id, mel_len, spk_embed

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len) :
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len) :
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def collate_tts(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]
    s_e = [x[4] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    s_e = torch.tensor(np.array(s_e))

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    return chars, mel, ids, mel_lens, s_e


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO : Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx) :
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


######################################################################################
# Speaker Encoder Dataset ############################################################
######################################################################################


# TODO: improve with a pool of speakers for data efficiency

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, hp.partials_n_frames) 


###########################################################################################
# Accent Style Encoder Dataset ############################################################
###########################################################################################
    
class AccentDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        spectrogram = np.load(item["path"])  # Load spectrogram as a NumPy array
        label = item["label"]

        # Add channel dimension for CNN
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Shape: (1, H, W)

        # Repeat the single channel to create 3 channels
        spectrogram = np.repeat(spectrogram, 3, axis=0)  # Shape: (3, H, W)

        # Apply transformations
        if self.transform:
            spectrogram = self.transform(torch.tensor(spectrogram, dtype=torch.float32))

        return spectrogram, label


def create_dataloaders(metadata, batch_size=32, test_size=0.2):
    
    # Encode labels as indices
    unique_labels = list(set(item["label"] for item in metadata))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # Update metadata with encoded labels
    for item in metadata:
        item["label"] = label_to_idx[item["label"]]

    # Split metadata into training and validation sets
    train_metadata, val_metadata = train_test_split(metadata, test_size=test_size, random_state=42)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VGG19 input size
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
    ])

    # Create datasets
    train_dataset = AccentDataset(train_metadata, transform=transform)
    val_dataset = AccentDataset(val_metadata, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes
    