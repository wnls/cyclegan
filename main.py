import os
import time

import argparse
import json

from visdom import Visdom
import numpy as np

import torch
import dataloader
from model.cycle_gan_model import CycleGANModel

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--unaligned', default=True, type=bool)
parser.add_argument('--img_size', default=256, type=bool)

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
parser.add_argument('--n_epoch', default=200, type=int)
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




if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {}...\n".format(device))

    # output files
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    file_format = os.path.join(args.out_dir, '{}_lr{}_wd{}_bs{:d}_ep{:d}'
                               .format(time.strftime("%m%d%H%M%S"), args.lr, args.wd, args.batch_size, args.n_epoch))
    # log_file = file_format + '.json'
    # checkpoint = file_format + '.pt'

    # load data
    if args.mode == "train":
        train_loader = dataloader.get_dataloader(args.train_A_dir, args.train_B_dir, batch_size=args.batch_size, unaligned=args.unaligned)
        val_loader = dataloader.get_dataloader(args.val_A_dir, args.val_B_dir, batch_size=args.batch_size, unaligned=args.unaligned)

    if args.mode == "test":
        test_loader = dataloader.get_dataloader(args.test_A_dir, args.test_B_dir, batch_size=args.batch_size, unaligned=args.unaligned)

    if args.vis:
        if args.port:
            viz = Visdom(port=int(args.port))
        else:
            viz = Visdom()

        startup_sec = 1
        while not viz.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
        assert viz.check_connection(), 'No connection could be formed quickly'

        win_train_G = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_train_D = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_train_tot = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_eval_G  = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_eval_D  = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_eval_tot  = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        # print('train window id =', win_train)
        # print('eval window id =', win_eval)
    else:
        viz = None

    model = CycleGANModel(args)

    # use pretrain
    start_epoch = 1
    if args.pretrain_path:
        print("Loading model from %s, mode: %s" % (args.pretrain_path, args.mode))
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

        train_iter = 0
        eval_iter = 0
        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            print("\n==== Epoch {:d} ====".format(epoch))

            # train
            for i, (A, B) in enumerate(train_loader):
                A, B = A.to(device), B.to(device)

                loss = model.train((A, B))

                # visualize train loss
                if viz:
                    viz.line(X=np.asarray([train_iter]), Y = np.asarray([loss['G_A']]), name='G_A', win=win_train_G)
                    viz.line(X=np.asarray([train_iter]), Y = np.asarray([loss['G_B']]), name='G_B', win=win_train_G)
                    viz.line(X=np.asarray([train_iter]), Y = np.asarray([loss['Cyc_A']]), name='Cyc_A', win=win_train_G)
                    viz.line(X=np.asarray([train_iter]), Y = np.asarray([loss['Cyc_B']]), name='Cyc_B', win=win_train_G)
                    viz.line(X=np.asarray([train_iter]), Y = np.asarray([loss['G']]), name='G', win=win_train_G)

                    viz.line(X=np.asarray([train_iter]), Y=np.asarray([loss['D_A']]), name='D_A', win=win_train_D)
                    viz.line(X=np.asarray([train_iter]), Y=np.asarray([loss['D_B']]), name='D_B', win=win_train_D)
                    viz.line(X=np.asarray([train_iter]), Y=np.asarray([loss['D']]), name='D', win=win_train_D)
                train_iter += 1

                # update stats
                s = ""
                for k, v in loss.items():
                    if stats['train_loss'].get(k) is None:
                        stats['train_loss'][k] = []
                    v = float(v)
                    stats['train_loss'][k].append(v)
                    s += "%s %.3f   " % (k, v)

                if i and i % args.print_every_train == 0:
                    print("Iter %d    loss %s" % (i, s))

            # eval
            print("\nEvaluating on val set...")
            # val_loss = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0, 'total': 0}
            for i, images in enumerate(val_loader):
                loss = model.eval(images)

                # visualize eval loss
                if viz:
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['G_A']]), name='G_A', win=win_eval_G)
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['G_B']]), name='G_B', win=win_eval_G)
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['Cyc_A']]), name='Cyc_A', win=win_eval_G)
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['Cyc_B']]), name='Cyc_B', win=win_eval_G)
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['G']]), name='G', win=win_eval_G)

                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['D_A']]), name='D_A', win=win_eval_D)
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['D_B']]), name='D_B', win=win_eval_D)
                    viz.line(X=np.asarray([eval_iter]), Y=np.asarray([loss['D']]), name='D', win=win_eval_D)
                eval_iter += 1

                s = ""
                for k, v in loss.items():
                    if stats['val_loss'].get(k) is None:
                        stats['val_loss'][k] = []
                    v = float(v)
                    stats['train_loss'][k].append(v)
                    s += "%s %.3f   " % (k, v)

                if i and i % args.print_every_val == 0:
                    print("Iter %d    loss %s" % (i, s))

            # print("Total val loss {:f}".format(i, val_loss['total']))

            # stats['val_loss']['G_A'].append(val_loss['G_A'])
            # stats['val_loss']['G_B'].append(val_loss['G_B'])
            # stats['val_loss']['D_A'].append(val_loss['D_A'])
            # stats['val_loss']['D_B'].append(val_loss['D_B'])
            # stats['val_loss']['total'].append(val_loss['total'])

            # save stats
            log_file = file_format + '_train.json'
            with open(log_file, "w") as f:
                json.dump(stats, f)

            # save model
            if epoch % args.save_every_epoch == 0:
                model_file = file_format + '_%d.pt' % (epoch)
                print("\nSaving model to %s\n" % (model_file))
                torch.save({'epoch': epoch, 'model_state': model.save_state()}, model_file)

        # save last epoch
        model_file = file_format + '_%d.pt' % (epoch)
        print("\nSaving model to %s\n" % (model_file))
        torch.save({'epoch': epoch, 'model_state': model.save_state()}, model_file)

    if args.mode == "test":
        print("\nEvaluating on test set...")
        test_loss = {'G_A': 0, 'G_B': 0, 'D_A': 0, 'D_B': 0, 'total': 0}
        for i, images in enumerate(test_loader):
            loss = model.eval(images)

            test_loss['G_A'] += loss['G_A']
            test_loss['G_B'] += loss['G_B']
            test_loss['D_A'] += loss['D_A']
            test_loss['D_B'] += loss['D_B']
            test_loss['total'] += loss['total']

        log_file = file_format + '_test.json'
        with open(log_file, "w") as f:
            json.dump(test_loss, f)




