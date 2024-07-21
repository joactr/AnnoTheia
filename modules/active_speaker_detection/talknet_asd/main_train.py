import warnings
warnings.filterwarnings("ignore")

import os
import glob
import time
import torch
import argparse

from torch.utils.data import DataLoader
from datasets.talknet_dataset import TalkNetDataset

from model.talknet_model import TalkNetModel

if __name__ == "__main__":

    # -- argument parser
    parser = argparse.ArgumentParser(
        description = "Script for the training & evaluation of TalkNet-ASD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- settings
    parser.add_argument("--device", default="cuda:0", type=str, help="Select the device.")
    parser.add_argument("--run-mode", default="training", type=str, help="Choose between 'training' or 'evaluation'")
    parser.add_argument("--load-model", default="./weights/english_talknetasd_ava_dataset.pth", type=str, help="Path to the checkpoint that has to be loaded into the model")

    # -- training details
    parser.add_argument('--learning-rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--lr-decay', default=0.95, type=float, help='Learning rate decay rate')
    parser.add_argument('--total-epochs', default=10, type=int, help='Maximum number of epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch Size')
    parser.add_argument('--window-size', default=25, type=int, help='Number of frames of input winfow')
    parser.add_argument('--n-workers', default=8, type=int, help='Number of loader threads')
    # -- dataset splits
    parser.add_argument("--training-dataset", default="", type=str, help="Path to where the training dataset split is")
    parser.add_argument("--validation-dataset", default="", type=str, help="Path to where the validation dataset split is")
    parser.add_argument("--test-dataset", default="", type=str, help="Path to where the test dataset split is")
    # -- output directories
    parser.add_argument('--output-dir', default="./talknet_exps/spanish/", type=str, help="Output directory where checkpoints and evaluations will be stored")
    args = parser.parse_args()

    # -- creating datasets
    train_dataset = TalkNetDataset(args.training_dataset, args.window_size)
    validation_dataset = TalkNetDataset(args.validation_dataset, args.window_size)
    test_dataset = TalkNetDataset(args.test_dataset, args.window_size)

    # -- defining data loaders
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.n_workers)
    validation_loader = DataLoader(dataset=validation_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.n_workers)


    # -- building and pre-training the TalkNet-ASD model
    talknet_asd = TalkNetModel(**vars(args));
    print(f"\nLoading TalkNet-ASD from checkpoint: {args.load_model}\n")
    talknet_asd.load_parameters(args.load_model)

    # -- training process
    if args.run_mode in ["training"]:
        # -- creating output directories
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

        mAPs = {"validation": [], "test": []}
        for epoch in range(1, args.total_epochs+1):
            model_output_filename = f"model_{args.window_size}frames_{str(epoch).zfill(3)}.pth"

            train_loss, lr = talknet_asd.train_network(loader=train_loader, epoch=epoch, **vars(args))

            # -- evaluation on validation & test
            val_output_path = os.path.join(args.output_dir, "results", f"validation_evaluated_from_{model_output_filename.split('.')[0]}.csv")
            val_loss, val_acc, val_mAP = talknet_asd.evaluate_network(loader = validation_loader, output_path=val_output_path, dataset="validation", **vars(args))
            mAPs["validation"].append(val_mAP)

            test_output_path = os.path.join(args.output_dir, "results", f"test_evaluated_from_{model_output_filename.split('.')[0]}.csv")
            test_loss, test_acc, test_mAP = talknet_asd.evaluate_network(loader = test_loader, output_path=test_output_path, dataset="test", **vars(args))
            mAPs["test"].append(test_mAP)

            print(f"Epoch {epoch}: TRAIN LOSS={round(train_loss, 3)} || VAL mAP={round(val_mAP, 3)}% (best: {max(mAPs['validation'])}%) || TEST mAP={round(test_mAP, 3)}% (best: {max(mAPs['test'])}%)")

            # -- saving epoch checkpoint
            checkpoint_output_path = os.path.join(
                args.output_dir,
                "checkpoints",
                f"model_{args.window_size}frames_{str(epoch).zfill(3)}.pth",
            )
            talknet_asd.save_parameters(checkpoint_output_path)

            print(f"\nPlease find the evaluated results in {os.path.join(args.output_dir, 'results')}")

    # -- evaluation process
    if args.run_mode in ["evaluation"]:
        # -- creating output directory
        os.makedirs(os.path.join(args.output_dir, "evaluation_results"), exist_ok=True)

        mAPs = {"validation": [], "test": []}
        model_output_filename = args.load_model.split(os.sep)[-1].split(".")[0]

        val_output_path = os.path.join(args.output_dir, "evaluation_results", f"validation_evaluated_from_{model_output_filename.split('.')[0]}.csv")
        val_loss, val_acc, val_mAP = talknet_asd.evaluate_network(loader = validation_loader, output_path=val_output_path, dataset="validation", **vars(args))
        mAPs["validation"].append(val_mAP)

        test_output_path = os.path.join(args.output_dir, "evaluation_results", f"test_evaluated_from_{model_output_filename.split('.')[0]}.csv")
        test_loss, test_acc, test_mAP = talknet_asd.evaluate_network(loader = test_loader, output_path=test_output_path, dataset="test", **vars(args))
        mAPs["test"].append(test_mAP)

        print(f"[EVALUATION] VAL mAP={round(val_mAP, 3)}% (best: {max(mAPs['validation'])}%) || TEST mAP={round(test_mAP, 3)}% (best: {max(mAPs['test'])}%)")
        print(f"\nPlease find the evaluated results in {os.path.join(args.output_dir, 'evaluation_results')}")
