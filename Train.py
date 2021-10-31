from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.utils.data as Data
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import argparse
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_name(0))

# define arg parser
def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument(
        '--folder', default='C:\\Users\\abc46\\Desktop\\NYCU\\VRDL\\hw1\\data\\train_data')
    parser.add_argument(
        '--label', default='C:\\Users\\abc46\\Desktop\\NYCU\\VRDL\\hw1\\data\\training_labels.txt')
    return parser


# define dataset
class Bird_dataset(Dataset):
    def __init__(self, root, label_file, transform):
        self.transform = transform
        images_path = Path(root)
        images_list = list(images_path.glob('*.jpg'))
        images_list_str = [str(x) for x in images_list]
        self.labels = {}
        file = open(label_file, 'r')
        for line in file.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            self.labels[k] = v
        file.close()
        self.images = images_list_str
    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        image = self.transform(image)
        label = int(self.labels[image_path[16:]][0:3])-1
        return image,label
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # define data preprocess
    train_transforms = transforms.Compose([
        transforms.Resize([400,400]),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  
        transforms.RandomRotation(45),
        transforms.RandomCrop(256),
        #transforms.ColorJitter(0.1,0.1,0),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # setting
    batch_size = 4
    epoch = 50
    num_train=3000
    num_classes=200
    initial_lr = 0.0002
    train_folder = args.folder
    train_label=args.label
    # prepare train data and val data
    train_bird_dataset = Bird_dataset(
        train_folder, train_label, train_transforms)
    train_set, val_set = torch.utils.data.random_split(
        train_bird_dataset, [int(0.95*len(train_bird_dataset)), int(0.05*len(train_bird_dataset))])
    train_loader = Data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=True)

    #model = models.resnet50(pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0','densenet201', pretrained=True)

    # pretrained model setting
    model = EfficientNet.from_pretrained(
        'efficientnet-b7', advprop=True)
    fc_inputs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Linear(fc_inputs, num_classes))
    if torch.cuda.is_available():
        model.cuda()

    '''optimizer = torch.optim.Adagrad(model.parameters(
                ), lr=initial_lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)'''
    '''optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)'''

    # define optimizer,learning rate scheduler,loss function
    optimizer = torch.optim.AdamW(model.parameters(),  lr=initial_lr, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=0.05, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    loss_func = nn.CrossEntropyLoss()

    #--------------------Training process----------------------------
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    the_last_loss = 100
    patience = 2
    trigger_times = 0
    for i in range(epoch):
        model.train()
        print('epoch {}'.format(i + 1))
        train_loss = 0.
        train_acc = 0.
        for img, label in train_loader:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_func(output, label)
            train_loss += loss.item()
            pred = torch.max(output, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss = float(train_loss / (0.95*num_train))
        Acc = float(train_acc / (0.95*num_train))
        loss_list.append(Loss)
        acc_list.append(Acc)
        print("Train Loss:%.9f" % Loss)
        print("Acc:%.9f" % Acc)

        val_loss = 0.
        val_acc = 0.
        model.eval()
        with torch.no_grad():
            for img, label in val_loader:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
                output = model(img)
                loss = loss_func(output, label)
                val_loss += loss.item()
                pred = torch.max(output, 1)[1]
                test_correct = (pred == label).sum()
                val_acc += test_correct.item()


        val_acc = float(val_acc / (0.05*num_train))
        val_loss = float(val_loss / (0.05*num_train))
        print("val Loss:%.9f" % val_loss)
        print("val_Acc:%.9f" % val_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        if val_loss > the_last_loss: 
            trigger_times += 1
            print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print('Early stopping!')
                break
        else:
            print('trigger times: 0')
            trigger_times = 0
            the_last_loss = val_loss

    #save model
    torch.save(model, './model/effnet-b7.pth')

    # plot results
    fig = plt.figure(figsize=(18, 6))
    fig.add_subplot(1, 2, 1)
    plt.plot(loss_list, label='Train_loss')
    plt.plot(val_loss_list, label='Val_loss')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid()
    plt.legend()

    fig.add_subplot(1, 2, 2)
    plt.plot(acc_list, label='Train_acc')
    plt.plot(val_acc_list, label='Val_acc')
    plt.grid()
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig('result.png')
    plt.show()
    print("finish!!")
