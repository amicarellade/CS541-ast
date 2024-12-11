import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import hparams as hp
import argparse

from utils.argutils import print_args
from pathlib import Path
from datasets import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from models.speaker_encoder import SpeakerEncoder
from utils.loggingutils import *
from pathlib import Path


def sync(device: torch.device):
    # FIXME
    return 
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        hp.speakers_per_batch,
        hp.utterances_per_speaker,
        num_workers=8,
    )
    
    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")
    
    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate_init)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = hp.learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    #vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    #vis.log_dataset(dataset)
    #vis.log_params()
    #device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    #vis.log_implementation({"Device": device_name})
    
    # Training loop
    #profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(loader, init_step):
        #profiler.tick("Blocking, waiting for batch (threaded)")
        bar = progbar(step, len(loader))
        
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        #profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        #profiler.tick("Forward pass")
        embeds_loss = embeds.view((hp.speakers_per_batch, hp.utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        #profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        loss.backward()
        #profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()
        #profiler.tick("Parameter update")
        
        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        #vis.update(loss.item(), eer, step)

        message = f'Batch progress: {bar} {step}/{len(loader)}' + " | Loss: {:.4f} | Error: {:.4f}".format(loss.item(), eer)
        stream(message)
        
        # Draw projections and save them to the backup folder
        """
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            #vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            #vis.save()
        """
        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print(" Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print(" Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        #profiler.tick("Extras (visualizations, saving)")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--run_id", default=hp.default_run_id, type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("--clean_data_root", default=os.path.join(hp.se_data_path, f"SV2TTS{hp.f_delim}encoder{hp.f_delim}"), type=Path, help= \
        "Path to the output directory of encoder_preprocess.py. If you left the default "
        "output directory when preprocessing, it should be <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default=hp.spk_encoder_model_path, help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-v", "--vis_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots.")
    parser.add_argument("-u", "--umap_every", type=int, default=100, help= \
        "Number of steps between updates of the umap projection. Set to 0 to never update the "
        "projections.")
    parser.add_argument("-s", "--save_every", type=int, default=500, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model.")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")
    parser.add_argument("--no_visdom", action="store_true", help= \
        "Disable visdom.")
    args = parser.parse_args()
    
    # Process the arguments
    args.models_dir.mkdir(exist_ok=True)
    
    # Run the training
    print_args(args, parser)
    train(**vars(args))
    