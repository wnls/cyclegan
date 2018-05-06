import os
from datetime import datetime

import argparse

import torch
import dataloader
from cycle_gan_model import CycleGANModel

parser = argparse.ArgumentParser()
# Model

# Training
parser.add_argument('--use_pretrain', default=0, type=int)
parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--print_every_train', default=50, type=int)
parser.add_argument('--print_every_val', default=100, type=int)
parser.add_argument('--save_every_epoch', default=10, type=int)
# Optimization
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--n_epoch', default=200, type=int)
parser.add_argument('--finetune', default=False, type=bool)
# Files
parser.add_argument('--out_dir', default='./checkpoints', type=str)
parser.add_argument('--train_A_dir', default='./datasets/maps/trainA', type=str)
parser.add_argument('--train_B_dir', default='./datasets/maps/trainB', type=str)
parser.add_argument('--val_A_dir', default='./datasets/maps/valA', type=str)
parser.add_argument('--val_B_dir', default='./datasets/maps/valB', type=str)




if __name__ == "__main__":
    args = parser.parse_args()

    # output files
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    now = datetime.now()
    file_format = os.path.join(args.outDir, '{}_lr{}_wd{}_bs{:d}_ep{:d}'
                                   .format(time.strftime("%m%d%H%M%S"), args.lr, args.wd, args.batch_size, args.n_epoch))
    log_file = file_format + '.json'
    # checkpoint = file_format + '.pt'

    # input files

    # load data
    if args.mode == "train":
        train_loader = dataloader.get_dataloader(args.train_A_dir, args.train_B_dir, args.batch_size)
        val_loader = dataloader.get_dataloader(args.val_A_dir, args.val_B_dir, args.batch_size)

    if args.mode == "test":
        # test_loader

    model = CycleGANModel(args)

    # use pretrain
    start_epoch = 1
    if args.pretrain_path:
        if args.mode == 'train' and args.use_pretrain:
            # TODO load GPU model on CPU
            checkpoint = torch.load(args.pretrain_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state(checkpoint['model_state'], train_mode=True)
        if args.mode == 'test':
            checkpoint = torch.load(args.pretrain_path)
            model.load_state(checkpoint['model_state'], train_mode=False)

    if (args.use_pretrain or args.mode == 'test') and args.pretrain_path:


    # TODO: GPU

    if args.mode == "train":
        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            print("\n\n==== Epoch {:d} ====".format(epoch))
            for i, images in enumerate(train_loader):

                model.train(images)

                # TODO update stats

            # save model
            if epoch % args.save_every_epoch == 0:
                torch.save({'epoch': epoch,
                            'model_state': model.save_state()}, file_format + '_{:d}.pt'.format(epoch))
        # save last epoch
        torch.save({'epoch': epoch,
                    'model_state': model.save_state()}, file_format + '_{:d}.pt'.format(epoch))

  if args.mode == "test":
    for i, images in enumerate(test_loader):
      model.test(images)





