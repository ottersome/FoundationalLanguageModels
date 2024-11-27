# %% Important imports
import argparse
from copy import deepcopy
from time import time
import yaml

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from transformers import AutoTokenizer, BartModel, BartTokenizer

from kgraphs.lightning.base_autoregressive import BaseAutoregressive
from kgraphs.models.models import Transformer
from kgraphs.utils.logging import create_logger
from kgraphs.dataprocessing.lightning import DocumentDataModule
import debugpy
from rich import traceback
traceback.install()

# %% Some global initalization
logger = create_logger("MAIN")


def argsies():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="manu/project_gutenberg")
    # Hyperparameters
    ap.add_argument("--epochs", default=10)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--pretrained_transformer_name", default="facebook/bart-base")
    ap.add_argument("--model_tokenwindow_size", default=1024)
    ap.add_argument("--model_dimension", default=768)
    ap.add_argument("--model_dimension_ff", default=3072)
    ap.add_argument("--num_layers", default=3)  # This is on the smaller side
    ap.add_argument("--num_heads", default=8)
    ap.add_argument("--dropout_rate", default=0.1)
    ap.add_argument("--masking_percentage", default=0.1)
    ap.add_argument("--raw_ds_location", default="./data/raw/")
    ap.add_argument("--seed", default=42,type=int)
    ap.add_argument("--stream", action="store_true", help="Whether or not to simmply stream all documents. (Local documents are not exhaustive")
    ap.add_argument("--stream_buffer_size", type=int, help="Size of buffer (unit: entire documetns) to keep before streaming more in.")

    ap.add_argument("--chkpnt_loc", default="./checkpoints", type=str)

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--training_steps", type=int)
    group.add_argument("--token_count_cap")

    ap.add_argument("-w", "--wandb", action="store_true")
    ap.add_argument("--wandb_project_name", help="Project name", type=str)
    ap.add_argument("--wr_name", help="Wand Run Name", type=str)
    ap.add_argument("--wr_notes", help="Wand Run Notes", type=str)

    ap.add_argument("--debug", action="store_true", help="Whether to debug using debugpy")
    ap.add_argument("--preferred_config", type=str, default="./configs/best_config.yaml")
    ap.add_argument("--split", type=float, nargs=3, default=[0.85, 0.15, 0])

    early_stopping = ap.add_argument_group("Early Stopping")
    early_stopping.add_argument("--early_stopping_steps", type=int, default=10)
    early_stopping.add_argument("--early_stopping_min_delta", type=float, default=0.001)

    args = ap.parse_args()

    ########################################
    # Important! 
    # We override the args with the yaml:
    # (So we can store training configs in a yaml file)
    ########################################
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    return args

def overload_parse_defaults_with_yaml(yaml_location:str, args: argparse.Namespace):
    with open(yaml_location, "r") as f:
        yaml_args = yaml.load(f, Loader=yaml.FullLoader)
        overloaded_args = recurse_til_leaf(yaml_args)
        for k, v in overloaded_args.items():
            if k in args.__dict__:
                # args.__dict__[k] = v
                # Change the property not they key
                setattr(args, k, v)
            else:
                raise ValueError(f"Key {k} not found in args")
    return args

def recurse_til_leaf(d: dict, parent_key: str = "") -> dict:
    return_dict = {}
    for k, v in d.items():
        next_key = f"{parent_key}_{k}" if parent_key != "" else k
        if isinstance(v, dict):
            deep_dict = recurse_til_leaf(v, parent_key=next_key)
            return_dict.update(deep_dict)
        else:
            return_dict[next_key] = v
    return return_dict

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





# %% Main Functions
if __name__ == "__main__":
    start_time = time()
    args = argsies()

    if args.debug:
        # We listen for this in 
        debugpy.listen(42020)
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()
        print("Client Connected.")

    # Initialize wandb
    if args.wandb:
        wandb.init(project="kgraphs")
    set_all_seeds(args.seed)

    # Load the Tokenizer
    tokenizer: BartTokenizer = AutoTokenizer.from_pretrained(args.pretrained_transformer_name)
    model_itself = BartModel.from_pretrained(args.pretrained_transformer_name)
    # Get only the embedding layer of this model
    embedding_layer = model_itself.get_input_embeddings()  # type: ignore
    for param in embedding_layer.parameters():
        param.requires_grad = False

    ########################################
    # Data Creation
    ########################################
    data_lmodule = DocumentDataModule( # Data Lightning Module
        args.dataset_name,
        args.stream_buffer_size,
        args.batch_size,
        args.model_tokenwindow_size,
        args.pretrained_transformer_name,
        args.split,
    )
    logger.info(f"Data Module for {args.dataset_name} is ready")

    # TODO: Change this with out own preference
    # val_dl = DataLoader(
    #     val_ds, batch_size=args.batch_size, drop_last=True, num_workers=1
    # )
    # test_it = next(iter(train_dl))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        args.model_dimension,
        args.num_heads,
        args.num_layers,
        args.model_dimension_ff,
        args.model_tokenwindow_size,
        args.dropout_rate,
        tokenizer.pad_token_id,
        embedding_layer,  # type: ignore
    ).to(device)

    # Output the amount of a parameters in the modelgru
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


    # Wrap it in the lightning Module
    lightning_module = BaseAutoregressive(model, tokenizer)

    # Get All arguments into a dictionary
    args_dict_str = str(deepcopy(vars(args)))
    notes = "The following are the training parameters\n"
    notes += args_dict_str

    # Initialize Wandb Logger
    wandb_logger = None
    if args.wandb:
        logger.info("🪄 Instantiating WandB")
        wandb_logger = WandbLogger(
            project=args.wandb_project_name, name=args.wr_name, notes=notes
        )

    ### Lightning Implemebtations
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.chkpnt_loc, save_top_k=3, monitor="val_loss"
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        accumulate_grad_batches=4,
        max_epochs=args.epochs,
        val_check_interval=0.05,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    # TODO: its having some problems right now
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(
    #     lightning_module,
    #     train_dataloaders=train_dl,
    #     val_dataloaders=val_dl,
    #     mode="binsearch",
    # )
    
    logger.info("Starting to train the model")
    # trainer.fit(lightning_module, train_dl, val_dl, ckpt_path="./checkpoints/epoch=0-step=2330.ckpt")
    trainer.fit(lightning_module, datamodule=data_lmodule)

    exit()

