import os

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive

import numpy as np

# new_labels = {idx: class_num for idx, class_num in enumerate(class_list)}
#     print("new_labels: ", new_labels)

class Cub2011(VisionDataset):

    base_folder = 'CUB_200_2011\\images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz"
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, class_list=[]):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.data = pd.DataFrame()
        self.class_list = class_list
        self.new_labels = {idx:val for idx, val in enumerate(self.class_list)}

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
        
        self.data = self.data[self.data.target.isin(self.class_list)]
            
        
        index_map = {val: idx for idx, val in enumerate(self.class_list)}
        self.data.target = [index_map[val] for val in self.data.target]
        
        

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True


    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')

        # download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
        #df = pd.DataFrame({'img': img, 'targets' : target}, index=range(len(self.data)))
        #return df




if __name__ == '__main__':   
    from cub2011 import Cub2011
    import os
    import torch 
    from torchvision.datasets.folder import default_loader
    from torchvision import transforms

    from options import args_parser

    
        
        
    
    args = args_parser()

    
    trans_cub_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    all_class_list = np.sort(np.random.choice(np.arange(1, 201), size=args.num_classes, replace=False))
    train_dataset = Cub2011('..\data', train=True, class_list=all_class_list, download=False, transform=trans_cub_train)
    test_dataset = Cub2011('../data', train=False, class_list=all_class_list, download=False,transform=trans_cub_train )

    print(test_dataset)

    
    # # 선택한 클래스 인덱스(0~199) 리스트 생성
    # selected_classes = [0, 2, 4, 6, 8] # 1,3,5,7,9 class
    
    # # 선택한 클래스에 해당하는 데이터 인덱스 리스트 생성
    # selected_indices = []
    # for i in range(len(train_dataset)):
    #     if train_dataset[i][1] in selected_classes:
    #         selected_indices.append(i)
    
    # # Subset 클래스를 사용하여 선택한 데이터셋으로 새로운 데이터셋 생성
    # train_dataset_selected = Subset(train_dataset, selected_indices)
    # targets_selected = [train_dataset_selected[i][1] for i in range(len(train_dataset_selected))]

    # print(train_dataset_selected)
    
    
    # classes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    # new_labels = {class_num: idx for idx, class_num in enumerate(classes)}
    
    # # # 데이터셋 내 라벨을 새로운 라벨로 재라벨링합니다.
    # for idx in range(len(train_dataset)):
    #     _, label = train_dataset[idx]
    #     if label in classes:
    #         train_dataset.data.iloc[idx, 2] = new_labels[label]
    #     else:
    #         train_dataset.data.iloc[idx, 2] = -1 # 선택한 10개의 클래스 이외의 클래스는 -1로 처리합니다.
        
    #print(np.unique(train_dataset.data.target, return_counts =True))
    
    
    
    
    
    
    
    
    
    
    
    
    # CUB_200_2011.tgz 위치
    #test_dataset = Cub2011('..\\..\\data', train=False, download=True)
    
    #testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2,
     #                                         shuffle=True, num_workers=2,drop_last=True)

    #print(train_dataset)                                          
    
    
    # # 첫 번째 미니배치를 가져옵니다.
    # data_iter = iter(trainloader)
    # images, labels = next(data_iter)

    # 첫 번째 미니배치의 크기를 출력합니다.
    # print(images.shape)
    # print(labels.shape)
    
    # for idx in range(len(train_dataset)):
    #     row = pd.DataFrame({'img': train_dataset[idx][0], 'targets': train_dataset[idx][1]}, index=[idx])
    #     df = pd.concat([df, row])

    #print(df)

    # # 데이터 추가
    # for i in range(len(train_dataset)):
    #     df = df.append({'img': train_dataset[i][0], 'targets': train_dataset[i][1]}, ignore_index=True)
    
    # print(df)
    
    
    # print(train_dataset)
    
    # images = pd.read_csv(os.path.join('.\\', 'CUB_200_2011', 'images.txt'), sep=' ',
    #                       names=['img_id', 'filepath'])
    # image_class_labels = pd.read_csv(os.path.join(".\\", 'CUB_200_2011', 'image_class_labels.txt'),
    #                                   sep=' ', names=['img_id', 'target'])
    # train_test_split = pd.read_csv(os.path.join(".\\", 'CUB_200_2011', 'train_test_split.txt'),
    #                                 sep=' ', names=['img_id', 'is_training_img'])

    # data = images.merge(image_class_labels, on='img_id')    
    # data = data.merge(train_test_split, on='img_id')
    # print(data)
     
    # class_names = pd.read_csv(os.path.join(".\\", 'CUB_200_2011', 'classes.txt'),
    #                            sep=' ', names=['class_name'], usecols=[1])
    # class_names = class_names['class_name'].to_list()
    # data = data[data.is_training_img == 1]
    # print(data)
     
    # sample = data
    # path = os.path.join(".\\", 'CUB_200_2011\\images', sample.filepath)
    # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
    # img = default_loader(path)
    # print(target)     
     
     
    # print(train_dataset[0])
    # print(len(train_dataset))
    # #print(train_dataset.target)