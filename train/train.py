import argparse
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # our hyperparameters all in a hparams dictionary
    parser.add_argument('--hparams', type=dict, default={
        # Required hparams
        "x_train": pd.DataFrame(),
        "x_val": pd.DataFrame(),
        "y_train": pd.DataFrame(),
        "y_val": pd.DataFrame(),
        # Optional hparams
        "backbone": "resnet34",
        "weights": "imagenet",
        "lr": 1e-3,
        "min_epochs": 6,
        "max_epochs": 1000,
        "patience": 4,
        "batch_size": 32,
        "num_workers": 0,
        "val_sanity_checks": 0,
        "fast_dev_run": False,
        "output_path": "model-outputs",
        "log_path": "tensorboard_logs",
        "gpu": 1,
    })
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--gpus', type=int, default=1) # could be used for multi-GPU as well
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--weights', type=str, default="imagenet")
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--min_epochs', type=int, default=6)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--val_sanity_checks', type=int, default=0)
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--log_path', type=str, default="tensorboard_logs")

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print(args)

    # Now we have all parameters and hyperparameters available and we need to match them with sagemaker 
    # structure. default_root_dir is set to out_put_data_dir to retrieve from training instances all the 
    # checkpoint and intermediary data produced by lightning
    mnistTrainer=pl.Trainer(hparams=args.hparams)

    # Set up our classifier class, passing params to the constructor
    model = MNISTClassifier(
        batch_size=args.batch_size, 
        train_data_dir=args.train, 
        test_data_dir=args.test
        )
    
    # Runs model training 
    ss_flood_model.fit(model) # orchestrates our model training

    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)