import os
import time

import argparse
import json
import math

import torch
import dataloader
from model.cycle_gan_model import CycleGANModel

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--unaligned', default=True, type=bool)
parser.add_argument('--resize', default=286, type=int)
parser.add_argument('--crop', default=256, type=int)
# Training
parser.add_argument('--mode', default="train", type=str)
parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--print_every_train', default=1, type=int)
parser.add_argument('--print_every_val', default=1, type=int)
parser.add_argument('--save_every_epoch', default=5, type=int)
# Optimization
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--n_epoch', default=1, type=int)
parser.add_argument('--finetune', default=False, type=bool)
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term of adam')
parser.add_argument('--lambda_A', default=10.0, type=float, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', default=10.0, type=float, help='weight for cycle loss (B -> A -> B)')
# Files
parser.add_argument('--out_dir', default='./checkpoints', type=str)
parser.add_argument('--train_A_dir', default='./datasets/maps/trainA', type=str)
parser.add_argument('--train_B_dir', default='./datasets/maps/trainB', type=str)
parser.add_argument('--val_A_dir', default='./datasets/maps/valA', type=str)
parser.add_argument('--val_B_dir', default='./datasets/maps/valB', type=str)
parser.add_argument('--test_A_dir', default='./datasets/maps/testA', type=str)
parser.add_argument('--test_B_dir', default='./datasets/maps/testB', type=str)



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {}...\n".format(device))

    args = parser.parse_args()
    for k, v in vars(args).items():
        print("%s = %s" % (k, v))

    # output files
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    file_format = os.path.join(args.out_dir, '{}_lr{}_wd{}_bs{:d}_ep{:d}'
                               .format(time.strftime("%m%d%H%M%S"), args.lr, args.wd, args.batch_size, args.n_epoch))
    print("\nSave model and stats file format = %s" % (file_format))

    # load data
    if args.mode == "train":
        train_loader = dataloader.get_dataloader(args.train_A_dir, args.train_B_dir, resize=args.resize, crop=args.crop, batch_size=args.batch_size, unaligned=args.unaligned)
        val_loader = dataloader.get_dataloader(args.val_A_dir, args.val_B_dir, resize=args.resize, crop=args.crop, batch_size=args.batch_size, unaligned=args.unaligned)

    if args.mode == "test":
        test_loader = dataloader.get_dataloader(args.test_A_dir, args.test_B_dir, resize=args.resize, crop=args.crop, batch_size=args.batch_size, unaligned=args.unaligned)

    model = CycleGANModel(args)

    # use pretrain
    start_epoch = 1
    if args.pretrain_path:
        print("\nLoading model from %s, mode: %s" % (args.pretrain_path, args.mode))
        if args.mode == 'train':
            # TODO load GPU model on CPU
            checkpoint = torch.load(args.pretrain_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state(checkpoint['model_state'])
        if args.mode == 'test':
            checkpoint = torch.load(args.pretrain_path)
            model.load_state(checkpoint['model_state'])

    model.to(device)

    if args.mode == "train":
        stats = {}
        stats['train_loss'] = {}
        stats['val_loss'] = {}
        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            print("\n==== Epoch {:d} ====".format(epoch))

            # train
            total_iter = math.ceil(len(train_loader) / args.batch_size)
            for i, (A, B) in enumerate(train_loader):
                A, B = A.to(device), B.to(device)

                loss = model.train((A, B))

                # update stats
                s = ""
                for k, v in loss.items():
                    if stats['train_loss'].get(k) is None:
                        stats['train_loss'][k] = []
                    v = round(float(v), 4)
                    stats['train_loss'][k].append(v)
                    s += "%s %f   " % (k, v)

                if i % args.print_every_train == 0:
                    print("Iter %d/%d    loss %s" % (i, total_iter, s))

            # eval
            print("\nEvaluating on val set...")
            total_iter = math.ceil(len(val_loader) / args.batch_size)
            total_val_loss = {}
            for i, images in enumerate(val_loader):
                loss = model.eval(images)

                s = ""
                for k, v in loss.items():
                    if stats['val_loss'].get(k) is None:
                        stats['val_loss'][k] = []
                    v = round(float(v), 4)
                    stats['val_loss'][k].append(v)
                    s += "%s %f   " % (k, v)

                    if total_val_loss.get(k) is None:
                        stats['val_loss'][k] = []
                    v = round(float(v), 4)
                    stats['val_loss'][k].append(v)

                if i % args.print_every_val == 0:
                    print("Iter %d/%d    loss %s" % (i, total_iter, s))

            # save stats
            log_file = file_format + '_train.json'
            with open(log_file, "w") as f:
                json.dump(stats, f)

            # save model
            if epoch % args.save_every_epoch == 0:
                model_file = file_format + '_%d.pt' % (epoch)
                print("\nSaving model to %s\n" % (model_file))
                torch.save({'epoch': epoch, 'model_state': model.save_state()}, model_file)

        # save model from last epoch
        model_file = file_format + '_%d.pt' % (epoch)
        print("\nSaving model to %s\n" % (model_file))
        torch.save({'epoch': epoch, 'model_state': model.save_state()}, model_file)

    if args.mode == "test":
        print("\nEvaluating on test set...")
        test_loss = {}
        for i, images in enumerate(test_loader):
            loss = model.eval(images)

            for k, v in loss.items():
                if test_loss.get(k) is None:
                    test_loss[k] = 0
                v = round(float(v), 4)
                test_loss[k] += v

        s = ""
        for k, v in test_loss.items():
            test_loss[k] = round(v / (i+1), 4)
            s += "%s %f   " % (k, test_loss[k])

        print("Average loss %s" % (s))

        log_file = file_format + '_test.json'
        with open(log_file, "w") as f:
            json.dump(test_loss, f)

