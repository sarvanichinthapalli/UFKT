
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os 
class Data:
    def __init__(self, gpu,data_dir,batch_size):
        pin_memory = False
        if gpu is not None:
            pin_memory = True

        scale_size = 224

        traindir = os.path.join(data_dir, 'ILSVRC2012_img_train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,#8,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,#8,
            pin_memory=True)

