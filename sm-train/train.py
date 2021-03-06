import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
import wandb
from datetime import datetime

from model import FloodModel


if __name__ =='__main__':
    
    # Removing this for Udacity, but useful for further usage
#     wandb.login() # This will look for WANDB_API_KEY env variable provided by secrets.env
#     wandb.init(project="Driven-Data-Floodwater-Mapping", entity="effective-altruism-techs")

    parser = argparse.ArgumentParser()
    
    # Below is only used when we are training one model.
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--architecture', type=str, default='Unet')
    parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--weights', type=str, default='imagenet')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_epochs', type=int, default=6)
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--val_sanity_checks', type=int, default=0)
    parser.add_argument('--log_path', type=str, default='tensorboard_logs')
    
    ## Below is only used when we are running a Hyperparameter Tuning job.
    
#     # hyperparameters sent by the client are passed as command-line arguments to the script.
#     parser.add_argument('--architecture', type=str, default=os.environ['SM_HP_ARCHITECTURE'])
#     parser.add_argument('--backbone', type=str, default=os.environ['SM_HP_BACKBONE'])
#     parser.add_argument('--weights', type=str, default=os.environ['SM_HP_WEIGHTS'])
#     parser.add_argument('--lr', type=float, default=os.environ['SM_HP_LR'])
#     parser.add_argument('--min_epochs', type=int, default=os.environ['SM_HP_MIN_EPOCHS'])
#     parser.add_argument('--max_epochs', type=int, default=os.environ['SM_HP_MAX_EPOCHS'])
#     parser.add_argument('--patience', type=int, default=os.environ['SM_HP_PATIENCE'])
#     parser.add_argument('--batch_size', type=int, default=os.environ['SM_HP_BATCH_SIZE'])
#     parser.add_argument('--num_workers', type=int, default=os.environ['SM_HP_NUM_WORKERS'])
#     parser.add_argument('--val_sanity_checks', type=int, default=os.environ['SM_HP_VAL_SANITY_CHECKS'])
#     parser.add_argument('--log_path', type=str, default=os.environ["SM_HP_LOG_PATH"])

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_s3_uri', type=str, default='s3://sagemaker-us-east-1-209161541854/floodwater_data')
    parser.add_argument('--train_features', type=str, default='s3://sagemaker-us-east-1-209161541854/floodwater_data/train_features')
    parser.add_argument('--train_labels', type=str, default='s3://sagemaker-us-east-1-209161541854/floodwater_data/train_labels')
    

    args, _ = parser.parse_known_args()
    print(args)
    
    seed_everything(9) # set a seed for reproducibility, seeds torch, numpy, python.random
    
    data_dir = "/opt/ml/input/data/data_s3_uri"
    
    # Read csv for training
    train_df = pd.read_csv(os.path.join(data_dir, "train_df.csv"))
    train_x = train_df[['chip_id', 'vv_path', 'vh_path']]
    train_y = train_df[['chip_id', 'label_path']]
    
    # Read csv for validation
    val_df = pd.read_csv(os.path.join(data_dir, "val_df.csv"))
    val_x = val_df[['chip_id', 'vv_path', 'vh_path']]
    val_y = val_df[['chip_id', 'label_path']]
    
    # Read csv for testing
    test_df = pd.read_csv(os.path.join(data_dir, "test_df.csv"))
    test_x = test_df[['chip_id', 'vv_path', 'vh_path']]
    test_y = test_df[['chip_id', 'label_path']]
        
    data_dict = {
        "x_train": train_x,
        "y_train": train_y,
        "x_val": val_x,
        "y_val": val_y,
        "x_test": test_x,
        "y_test": test_y,
    }
    
    hparams = {
        # Optional hparams, set these in the hparams dictionary in the main notebook before training
        "architecture": args.architecture,
        "backbone": args.backbone,
        "weights": args.weights,
        "lr": args.lr,
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_sanity_checks": args.val_sanity_checks,
        "fast_dev_run": False,
        "output_path": "model-outputs",
        "log_path": "tensorboard_logs",
        "gpu": True,
    }
    hparams.update(data_dict)
    print(hparams)
    
    available_gpus = torch.cuda.is_available()
    
    print("Is there an available GPU? ", available_gpus)
    print("Device count: ", torch.cuda.device_count())

    # Set up our classifier class, passing params to the constructor
    ss_flood_model = FloodModel(hparams=hparams)
    
    # Runs model training 
    ss_flood_model.fit() # orchestrates our model training
    
    best_model = FloodModel(hparams=hparams)
    best_model_path = ss_flood_model.trainer_params["callbacks"][0].best_model_path
    best_model = best_model.load_from_checkpoint(checkpoint_path=best_model_path)
    
    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(best_model.state_dict(), f)
        
#     wandb.finish()