import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class GANDataset(Dataset):

	# Initial logic here, including reading the image files and transform the data
	def __init__(self, rootA, rootB, transform=None):
		# initialize image path and transformation
		self.image_pathsA = list(map(lambda x: os.path.join(rootA, x), os.listdir(rootA)))
		self.image_pathsB = list(map(lambda x: os.path.join(rootB, x), os.listdir(rootB)))
		self.transform = transform

	# read the image and preprocess
	def __getitem__(self, index):
		image_pathA = self.image_pathsA[index]
		image_pathB = self.image_pathsB[index]

		imageA = Image.open(image_pathA).convert('RGB')
		imageB = Image.open(image_pathB).convert('RGB')

		if self.transform is not None:
			imageA = self.transform(imageA)
			imageB = self.transform(imageB)

		return imageA, imageB

	# returns the number of examples we read
	def __len__(self):
		return len(self.image_pathsA)  # of how many examples we have


## return - DataLoader
def get_dataloader(image_pathA, image_pathB, batch_size, image_size=(600, 600)):
	transform = transforms.Compose([
		# resize PIL image to given size
		transforms.Resize(image_size),
		# crop image at randomn location
		transforms.RandomCrop(image_size),
		# flip images randomly
		transforms.RandomHorizontalFlip(),
		# convert image input into torch tensor
		transforms.ToTensor(),
		# normalize image with mean and standard deviation
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	batch_dataset = GANDataset(image_pathA, image_pathB, transform)

	#     B_dataloader = DataLoader(dataset=dataset['B'], batch_size=batch_size, shuffle=True)

	return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=True)

