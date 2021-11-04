import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
import os
import torch.utils.data as Data
from PIL import Image
from torch.autograd import Variable
import argparse
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_name(0))


# define arg parser
def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument(
        '--folder', default='C:\\Users\\abc46\\Desktop\\NYCU\\VRDL\\hw1\\data\\test_data')
    parser.add_argument(
        '--labelmap', default='C:\\Users\\abc46\\Desktop\\NYCU\\VRDL\\hw1\\data\\classes.txt')
    parser.add_argument(
        '--order', default='C:\\Users\\abc46\\Desktop\\NYCU\\VRDL\\hw1\\data\\testing_img_order.txt')
    parser.add_argument(
        '--model', default='C:\\Users\\abc46\\Desktop\\NYCU\\VRDL\\hw1\\model\\effnet-b7.0.0002_50_adamW_advprop.pth')
    return parser


# define dataset
class Bird_dataset(Dataset):
    def __init__(self, root, folder, transform):
        self.transform = transform
        with open(root) as f:
            test_images = [x.strip() for x in f.readlines()]  # all the testing images
        self.images = test_images
        self.folder = folder

    def __getitem__(self, item):
        image_path = self.folder+"\\"+self.images[item]
        image = Image.open(image_path)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    test_folder = args.folder
    test_order = args.order
    label_map = args.labelmap
    model_path = args.model
    # load model file
    model = torch.load(model_path)
    # define data preprocess
    test_transforms = transforms.Compose([
        transforms.Resize([400, 400]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # prepare test data(pictures)
    test_bird_dataset = Bird_dataset(
        test_order, test_folder, test_transforms)
    test_loader = Data.DataLoader(
        dataset=test_bird_dataset, batch_size=1, shuffle=False)

    # prepare label map
    labels = {}
    file = open(label_map, 'r')
    for line in file.readlines():
        line = line.strip()
        k = int(line.split('.')[0])-1
        v = line
        labels[k] = v
    file.close()

    # predicting
    model.eval()
    submission = []
    num = 0
    with torch.no_grad():
        print("predicting......")
        for img in test_loader:  # image order is important to your result
            img = Variable(img).cuda()
            output = model(img)  # the predicted category
            pred = torch.max(output, 1)[1]
            pred_ans = labels[int(pred)]
            submission.append([test_bird_dataset.images[num], pred_ans])
            num = num+1
    # save result
    np.savetxt('answer.txt', submission, fmt='%s')
    print("predicting finish!")
