from torch.utils.data import Dataset, DataLoader

class GANDataset(Dataset):
	def __init__(self):
	    

  	def __getitem__(self, index):
  	
  	
  	def __len__(self):


## return - DataLoader
 def get_dataloader(args):

 	return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    	