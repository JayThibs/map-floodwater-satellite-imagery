import argparse
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=1) # used to support multi-GPU or CPU training

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr','--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('-te','--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print(args)

    # Now we have all parameters and hyperparameters available and we need to match them with sagemaker 
    # structure. default_root_dir is set to out_put_data_dir to retrieve from training instances all the 
    # checkpoint and intermediary data produced by lightning
    mnistTrainer=pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, default_root_dir=args.output_data_dir)

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