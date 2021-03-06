import os
import time

import argparse
import json
import math

from visdom import Visdom
import numpy as np

import torch
import dataloader
from model.cycle_gan_model import CycleGANModel

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--unaligned', default=True, type=bool)
parser.add_argument('--resize', default=286, type=int)
parser.add_argument('--crop', default=256, type=int)
parser.add_argument('--G', default='res6', type=str, help='res6|res9|unet')
parser.add_argument('--D', default='vanilla', type=str, help='vanilla|dual|deep|deepdual')
parser.add_argument('--concat', default=True, action='store_false', help='concatenate or add in skip connection')
# Training
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--mode', default="train", type=str, help='train|test|gen-val|gen-train')
parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--print_every_train', default=100, type=int)
parser.add_argument('--print_every_val', default=200, type=int)
parser.add_argument('--save_every_epoch', default=20, type=int)
parser.add_argument('--eval_n', default=100, type=int, help='number of examples from val set to evaluate on each epoch')
parser.add_argument('--save_n_img', default=10000, type=int, help='number of images to save at test time')
parser.add_argument('--test_collage', default='single', type=str, help='single|basic|idt. what to output at test')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--init_type', default='normal', type=str, help='normal|xavier|kaiming. initialization for weights for G and D')
parser.add_argument('--suffix', default='', type=str, help='out dir suffix')
# Optimization
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--lr_decay_start', default=100, type=int, help='eppch to start lr decay')
parser.add_argument('--lr_decay_n', default=100, type=int, help='number of epochs to decay lr to 0')
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--n_epoch', default=100, type=int)
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term of adam')
parser.add_argument('--lambda_A', default=10.0, type=float, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', default=10.0, type=float, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_idt', default=0.5, type=float)
parser.add_argument('--lambda_D', default=0.5, type=float, help='D scale')
# Files
parser.add_argument('--out_dir', default='./checkpoints', type=str)
parser.add_argument('--data_dir', default='./datasets/maps/', type=str)

# Visualization
parser.add_argument('--vis', default=False, action='store_true')
parser.add_argument('--port', default=8097, type=int)


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda:%d" % args.device_id if torch.cuda.is_available() else "cpu")

    s = "Using %s\n\n" % device
    for k, v in vars(args).items():
        s += "%s = %s\n" % (k, v)
    print(s)

    # output files
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "train":
        out_dir = os.path.join(args.out_dir, "%s%s" % (time.strftime("%m%d%H%M%S"),
                                                       "_"+args.suffix if len(args.suffix)!=0 else ""))
        os.mkdir(out_dir)
        out_dir_img = os.path.join(out_dir, "images")
        os.mkdir(out_dir_img)
        log_file = os.path.join(out_dir, "train.json")
        config_file = os.path.join(out_dir, "config.txt")

        # save configs to config file
        with open(config_file, "w") as f:
            f.write(s)
        print("\nSave model and stats to directory %s" % (out_dir))

        # load data
        train_loader = dataloader.get_dataloader(os.path.join(args.data_dir, "trainA"),
                                                 os.path.join(args.data_dir, "trainB"),
                                                 resize=args.resize, crop=args.crop,
                                                 batch_size=args.batch_size, unaligned=args.unaligned,
                                                 device=device, num_workers=args.num_workers)
        val_loader = dataloader.get_dataloader(os.path.join(args.data_dir, "valA"),
                                               os.path.join(args.data_dir, "valB"),
                                               resize=args.resize, crop=args.crop,
                                               batch_size=1, unaligned=args.unaligned, #TODO val batch size
                                               device=device, num_workers=args.num_workers, test=True)
    if args.mode == "test":
        out_dir = os.path.dirname(args.pretrain_path)
        out_dir_img = os.path.join(out_dir, "images", "test")
        try:
            os.mkdir(out_dir_img)
        except OSError:
            pass

        # load data
        test_loader = dataloader.get_dataloader(os.path.join(args.data_dir, "testA"),
                                                os.path.join(args.data_dir, "testB"),
                                                resize=args.resize, crop=args.crop,
                                                batch_size=1, unaligned=args.unaligned,
                                                device=device, num_workers=args.num_workers, shuffle=False, test=True)
    if args.mode == "gen-train":
        out_dir = os.path.dirname(args.pretrain_path)
        out_dir_img = os.path.join(out_dir, "images", "train")
        try:
            os.mkdir(out_dir_img)
        except OSError:
            pass

        # load data
        test_loader = dataloader.get_dataloader(os.path.join(args.data_dir, "trainA"),
                                                os.path.join(args.data_dir, "trainB"),
                                                resize=args.resize, crop=args.crop,
                                                batch_size=1, unaligned=args.unaligned,
                                                device=device, num_workers=args.num_workers, shuffle=False, test=True)
    if args.mode == "gen-val":
        out_dir = os.path.dirname(args.pretrain_path)
        out_dir_img = os.path.join(out_dir, "images", "val")
        try:
            os.mkdir(out_dir_img)
        except OSError:
            pass

        # load data
        test_loader = dataloader.get_dataloader(os.path.join(args.data_dir, "valA"),
                                                os.path.join(args.data_dir, "valB"),
                                                resize=args.resize, crop=args.crop,
                                                batch_size=1, unaligned=args.unaligned,
                                                device=device, num_workers=args.num_workers, shuffle=False, test=True)

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

        win_train_G = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Train Loss (Genarator + Cycle)', 'showlegend': False})
        win_train_D = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Train Loss (Discriminator)', 'showlegend': False})
        win_train_total = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Train Loss (Total)', 'showlegend': False})

        win_eval_G = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Eval Loss (Genarator + Cycle)', 'showlegend': False})
        win_eval_D = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Eval Loss (Discriminator)', 'showlegend': False})
        win_eval_total = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Eval Loss (Total)', 'showlegend': False})

        win_comp_tot = viz.line(X=np.asarray([0]), Y=np.asarray([0]), opts={'title': 'Loss', 'showlegend': False})

        # print('train window id =', win_train)
        # print('eval window id =', win_eval)
    else:
        viz = None

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

    model.set_start_epoch(start_epoch)
    model.to(device)

    if args.mode == "train":
        stats = {}
        stats['train_loss'] = {}
        stats['val_loss'] = {}

        total_train_iter = len(train_loader)
        eval_n = min(args.eval_n, len(val_loader)) #TODO batch
        total_val_iter = eval_n

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            print("\n==== Epoch {:d} ====".format(epoch))
            t_start = time.time()

            # train
            for i, images in enumerate(train_loader):

                loss = model.train(images, save=(i == 0), out_dir_img=out_dir_img, epoch=epoch)

                # update stats
                s = ""
                for k, v in loss.items():
                    if stats['train_loss'].get(k) is None:
                        stats['train_loss'][k] = []
                    # convert Tensor to float
                    v = round(float(v), 4)
                    stats['train_loss'][k].append(v)
                    loss[k] = v
                    s += "%s %f   " % (k, v)

                if i % args.print_every_train == 0:
                    print("Iter %d/%d    loss %s" % (i, total_train_iter, s))

                # visualize train loss
                iter_id = epoch + i / len(train_loader)
                if args.vis:
                    opt = {'X': np.asarray([iter_id]), 'opts': {'showlegend': True}, 'update': 'append'}

                    opt['win'] = win_train_G
                    for item in ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'G']:
                        opt['Y'] = np.asarray([loss[item]])
                        opt['name'] = item
                        viz.line(**opt)

                    opt['win'] = win_train_D
                    for item in ['D_A', 'D_B', 'D']:
                        opt['Y'] = np.asarray([loss[item]])
                        opt['name'] = item
                        viz.line(**opt)

                    opt['win'] = win_train_total
                    for item in ['D', 'G']:
                        opt['Y'] = np.asarray([loss[item]])
                        opt['name'] = item
                        viz.line(**opt)

                    if i == len(train_loader) - 1:
                        viz.line(X=np.asarray([epoch]), Y=np.asarray([loss['G']]), name='Train G', win=win_comp_tot, opts={'showlegend': True}, update='append')
                        viz.line(X=np.asarray([epoch]), Y=np.asarray([loss['D']]), name='Train D', win=win_comp_tot, opts={'showlegend': True}, update='append')

            print("Time taken: %.2f m" % ((time.time() - t_start) / 60))

            # eval
            if eval_n > 0:
                print("\nEvaluating %d examples on val set..." % eval_n)
                total_val_loss = {}
                avg_val_loss = {}
                for i, images in enumerate(val_loader):
                    if i >= args.eval_n:
                        i -= 1
                        break

                    loss = model.eval(images, save=(i == 0), out_dir_img=out_dir_img, epoch=epoch)

                    # update stats
                    s = ""
                    for k, v in loss.items():
                        if stats['val_loss'].get(k) is None:
                            stats['val_loss'][k] = []
                        if total_val_loss.get(k) is None:
                            total_val_loss[k] = 0
                        # convert Tensor to float
                        v = round(float(v), 4)
                        stats['val_loss'][k].append(v)
                        total_val_loss[k] += v
                        loss[k] = v
                        s += "%s %f   " % (k, v)

                    if i % args.print_every_val == 0:
                        print("Iter %d/%d    loss %s" % (i, total_val_iter, s))


                # calculate avg val loss
                s = ""
                for k, v in total_val_loss.items():
                    avg_val_loss[k] = v / (i + 1)
                    s += "%s %f   " % (k, v / (i + 1))
                print("Average val loss    %s" % s)

                # visualize val loss
                if args.vis:
                    opt = {'X': np.asarray([epoch]), 'opts': {'showlegend': True}, 'update': 'append'}

                    opt['win'] = win_eval_G
                    for item in ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'G']:
                        opt['Y'] = np.asarray([avg_val_loss[item]])
                        opt['name'] = item
                        viz.line(**opt)

                    opt['win'] = win_eval_D
                    for item in ['D_A', 'D_B', 'D']:
                        opt['Y'] = np.asarray([avg_val_loss[item]])
                        opt['name'] = item
                        viz.line(**opt)

                    opt['win'] = win_eval_total
                    for item in ['D', 'G']:
                        opt['Y'] = np.asarray([avg_val_loss[item]])
                        opt['name'] = item
                        viz.line(**opt)

                    viz.line(X=np.asarray([epoch]), Y=np.asarray([avg_val_loss['G']]), name='Val G', win=win_comp_tot, opts={'showlegend': True}, update='append')
                    viz.line(X=np.asarray([epoch]), Y=np.asarray([avg_val_loss['D']]), name='Val D', win=win_comp_tot, opts={'showlegend': True}, update='append')
                    
            # save stats
            with open(log_file, "w") as f:
                json.dump(stats, f)

            # save model
            if epoch % args.save_every_epoch == 0:
                model_file = os.path.join(out_dir, "epoch_%d.pt" % epoch)
                # model_file = file_format + '_%d.pt' % (epoch)
                print("\nSaving model to %s\n" % (model_file))
                torch.save({'epoch': epoch, 'model_state': model.save_state()}, model_file)

            # update scheduler
            model.update_scheduler()

        # save model from last epoch
        model_file = os.path.join(out_dir, "epoch_%d.pt" % epoch)
        # model_file = file_format + '_%d.pt' % (epoch)
        print("\nSaving model to %s\n" % (model_file))
        torch.save({'epoch': epoch, 'model_state': model.save_state()}, model_file)

    if args.mode == "test":
        print("\nEvaluating on test set...")
        for i, images in enumerate(test_loader):
            if i >= args.save_n_img:
                break
            model.test(images, i, out_dir_img, args.test_collage)

        # test_loss = {}
        # for i, images in enumerate(test_loader):
        #     loss = model.eval(images)
        #
        #     for k, v in loss.items():
        #         if test_loss.get(k) is None:
        #             test_loss[k] = 0
        #         v = round(float(v), 4)
        #         test_loss[k] += v
        #
        # s = ""
        # for k, v in test_loss.items():
        #     test_loss[k] = round(v / (i+1), 4)
        #     s += "%s %f   " % (k, test_loss[k])
        #
        # print("Average loss %s" % (s))
        #
        # log_file = os.path.join(out_dir, "test.json")
        # with open(log_file, "w") as f:
        #     json.dump(test_loss, f)

