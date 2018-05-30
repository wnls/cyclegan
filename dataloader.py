import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class GANDataset(Dataset):

    # Initial logic here, including reading the image files and transform the data
    def __init__(self, rootA, rootB, transform=None, unaligned=False, device=None, test=False):
        # initialize image path and transformation
        sortedA = sorted(os.listdir(rootA), key=lambda name: int(name.split('_')[0]))
        sortedB = sorted(os.listdir(rootB), key=lambda name: int(name.split('_')[0]))
        self.image_pathsA = list(map(lambda x: os.path.join(rootA, x), sortedA))
        self.image_pathsB = list(map(lambda x: os.path.join(rootB, x), sortedB))

        self.transform = transform
        self.unaligned = unaligned
        self.device = device
        self.test = test

    # override to support indexing
    def __getitem__(self, index):
        
        image_pathA = self.image_pathsA[index]
        imageA = Image.open(image_pathA).convert('RGB')

        # unaligned the paired images if needed
        index_B = index
        if self.unaligned:
            index_B = random.randint(0, len(self.image_pathsB)-1)
            image_pathB = self.image_pathsB[index_B]
        else:
            image_pathB = self.image_pathsB[index]
            
        imageB = Image.open(image_pathB).convert('RGB')

        imageA_gt = Image.open(self.image_pathsB[index]).convert('RGB')
        imageB_gt = Image.open(self.image_pathsA[index_B]).convert('RGB')
        
        # transform the images if needed
        if self.transform is not None:
            if self.test:
                imageA = self.transform(imageA)
                imageA_gt = self.transform(imageA_gt)
                imageB = self.transform(imageB)
                imageB_gt = self.transform(imageB_gt)
            else: # if test is False, need to crop and flip source and gt the same way
                seed = random.randint(0, 2147483647)
                random.seed(seed)
                imageA = self.transform(imageA)
                random.seed(seed)
                imageA_gt = self.transform(imageA_gt)

                seed = random.randint(0, 2147483647)
                random.seed(seed)
                imageB = self.transform(imageB)
                random.seed(seed)
                imageB_gt = self.transform(imageB_gt)

        # convert to GPU tensor
        if self.device is not None:
            imageA = imageA.to(self.device)
            imageB = imageB.to(self.device)
            imageA_gt = imageA_gt.to(self.device)
            imageB_gt = imageB_gt.to(self.device)

        return imageA, imageB, imageA_gt, imageB_gt, index+1, index_B+1

    # returns the number of examples we read
    def __len__(self):
        return max(len(self.image_pathsA), len(self.image_pathsB))  # of how many examples we have


## return - DataLoader, batch dimension in (batch_size, channel, H, W)
def get_dataloader(image_pathA, image_pathB, batch_size, resize, crop, unaligned=False, device=None, shuffle=True, test=False, num_workers=0):
    if test:
        transform = transforms.Compose([
            transforms.Resize(crop, Image.BICUBIC), # resize to crop size directly
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    else:
        transform = transforms.Compose([
            # resize PIL image to given size
            transforms.Resize(resize, Image.BICUBIC),
            # crop image at randomn location
            transforms.RandomCrop(crop),
            # flip images randomly
            transforms.RandomHorizontalFlip(),
            # convert image input into torch tensor
            transforms.ToTensor(),
            # normalize image with mean and standard deviation
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_dataset = GANDataset(image_pathA, image_pathB, transform, unaligned, device, test)

    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

