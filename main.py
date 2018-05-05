import os

import argparse

import dataloader
from cycle_gan_model import CycleGANModel

parser = argparse.ArgumentParser()
# Model

# Training
parser.add_argument('--use_pretrain', default=0, type=int)
parser.add_argument('--pretrained_path', default='', type=str)
parser.add_argument('--print_every_train', default=50, type=int)
parser.add_argument('--print_every_val', default=100, type=int)
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
	if not os.path.exists(args.outDir):
		os.mkdir(args.outDir)
  	now = datetime.now()
  	file_format = os.path.join(args.outDir, 'lr{}_wd{}_bts{:d}_ep{:d}_{}'
  		.format(args.lr, args.wd, args.batch_size, args.n_epoch, time.strftime("%m%d%H%M%S")))
  	log_file = file_format + '.json'
  	checkpoint = file_format + '.pt'

  	# input files

  	# load data
  	if args.mode == "train":
  		train_loader = dataloader.get_dataloader(args.train_A_dir, ars.train_B_dir, args.batch_size)
  		val_loader = dataloader.get_dataloader(args.val_A_dir, ars.val_B_dir, args.batch_size)
  		
  	if args.mode == "test":
  		# test_loader

  	model = CycleGANModel(args)
  	# loss = 
	# optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

  	# use pretrain
  	if args.use_pretrain and args.pretrained_path:
	    print('Using pretrained model', args.pretrained_path)
	    pretrained = torch.load(args.pretrained_path)
	    model.load_state_dict(pretrained['model'])
	    optim.load_state_dict(pretrained['optim'])
	    # set model lr to new lr
	    for param_group in optim.param_groups:
	      before = param_group['lr']
	      param_group['lr'] = args.lr
	      print('optim lr: before={} / after={}'.format(before, args.lr))		

	# TODO: GPU

	if args.mode == "train":
		for e in range(args.n_epoch):
			print("\n\n==== Epoch {:d} ====".format(e+1))
			for i, images in enumerate(loader):

      			model.train(images)
      			# update stats

      		# save model
          
  if args.mode == "test":
    for i, images in enumerate(loader):
      model.test(images)





