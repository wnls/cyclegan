import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./checkpoints/0515091105_no idt_bs2/images/', type=str)
parser.add_argument('--generated', default=False, action='store_true')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--eval_img', default='target', type=str)
parser.add_argument('--img_ext', default='png', type=str)
parser.add_argument('--img_row', default=1, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--splits', default=1, type=int)

"""customized data loading"""
def Gen_dataProcessing(image, img_size = 256, eval_img='target', img_row=1):
    # images should be Pil format
    # width, height = image.size
    row = img_row

    if eval_img == 'target':
        left = img_size
        top = (row-1)*img_size
        right = 2 * img_size
        bottom = row*img_size
    elif eval_img == 'source':
        left = 0
        top = (row-1)*img_size
        right = img_size
        bottom = row*img_size
    elif eval_img == 'cycle':
        left = 2 * img_size
        top = (row-1)*img_size
        right = 3 * img_size
        bottom = row*img_size
    else:
        raise Exception("choose target, source, or cycle")

    cropped_example = image.crop((left, top, right, bottom))

    return cropped_example

class GenDataset(Dataset):

    # Initial logic here, including reading the image files and transform the data
    def __init__(self, root, transform=None, device=None, generated=True,
                 eval_img='target', img_ext='png', img_size=256, img_row=1):
        # initialize image path and transformation
        def list_files(directory, extension):
            return (f for f in os.listdir(directory) if f.endswith('.' + extension))

        self.image_path = list(map(lambda x: os.path.join(root, x), list_files(root,img_ext)))
        # self.image_path = list(map(lambda x: os.path.join(root, x) if x.endswith('.png') else False, os.listdir(root)))
        self.transform = transform
        self.device = device
        self.generated = generated
        self.eval_img = eval_img
        self.img_size = img_size
        self.img_row = img_row

    # override to support indexing
    def __getitem__(self, index):

        image_path = self.image_path[index]
        image = Image.open(image_path)

        # process aggregated images if needed
        if self.generated:
            image = Gen_dataProcessing(image, img_size=self.img_size, eval_img=self.eval_img, img_row=self.img_row)

        # transform the images if needed
        if self.transform is not None:
            image = self.transform(image)

        # convert to GPU tensor
        if self.device is not None:
            image = image.to(self.device)

        return image

    # returns the number of examples we read
    def __len__(self):
        return len(self.image_path)  # of how many examples we have

## return - DataLoader, batch dimension in (batch_size, channel, H, W)
def get_dataloader(image_path, batch_size, generated=True, eval_img='target', img_ext='png',
                   img_size=256, img_row=1, device=None, num_workers=0):
    transform = transforms.Compose([
        # resize PIL image to given size
        transforms.Resize(299, Image.BICUBIC),
        # convert image input into torch tensor
        transforms.ToTensor(),
        # normalize image with mean and standard deviation
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_dataset = GenDataset(image_path, transform, device, generated, eval_img, img_ext, img_size, img_row)

    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

"""calculate inception score"""
def inception_score(image_folder, batch_size=10, generated=True, eval_img='target',
                    img_ext='png', img_size=256, img_row=1, splits=1):
    """Computes the inception score of the generated images
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    # images should be in Torch dataset format in the follow dimension C, H, W, normalized
    dataloader = get_dataloader(image_folder, batch_size, generated=generated, eval_img=eval_img,
                                img_ext=img_ext, img_size=img_size, img_row=img_row)
    classifier = inception_v3(pretrained=True, transform_input=False)

    N = len(os.listdir(image_folder))
    if generated:
        N -= 1
    print('the number of the images are', N)

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        print('You do not have a GPU available, model is running on CPU!')
        dtype = torch.FloatTensor

    # change inception model to evaluation mode
    classifier = classifier.type(dtype)
    classifier.eval()

    # Get predictions
    predictions = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        # print(batch.size()) # check the dimension of a batch
        batch = batch.type(dtype)
        b_size = batch.size()[0]
        batch_score = classifier(batch)
        predictions[i*batch_size : i*batch_size+b_size] = F.softmax(batch_score,dim=1).data.cpu().numpy()

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = predictions[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':

    args = parser.parse_args()

    print ("Calculating Inception Score...")
    print (inception_score(args.data_dir,
                           batch_size=args.batch_size,
                           generated=args.generated,
                           eval_img=args.eval_img,
                           img_ext=args.img_ext,
                           img_size=args.img_size,
                           img_row=args.img_row,
                           splits=args.splits))


