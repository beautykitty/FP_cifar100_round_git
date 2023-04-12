#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
#
import numpy as np

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



# class_list 추가
def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    ### 
    global_protos = []
    global_protos2 = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy, test_accuracy, test_accuracy_wo = [], [], [], []
    train_acc_user = []
    test_acc_user = []
    test_acc_user_wo = []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos, local_protos2 = [], [], {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        ###
        proto_loss = 0
        proto2_loss = 0
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            ###
            w, loss, acc, protos, protos2 = local_model.update_weights_het(args, idx, global_protos, global_protos2, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            ###
            agg_protos = agg_func(protos)
            agg_protos2 = agg_func(protos)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            ###
            local_protos[idx] = agg_protos
            local_protos2[idx] = agg_protos 
            
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            
            ###
            proto_loss += loss['2']
            proto2_loss += loss['3']
            
            #
            train_acc_user.append(acc)
            
        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        ###
        global_protos = proto_aggregation(local_protos)
        global_protos2 = proto_aggregation(local_protos2)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        train_acc_user_avg = sum(train_acc_user)/len(train_acc_user)
        train_accuracy.append(train_acc_user_avg)
        
        
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt2(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, global_protos2)
        
        test_acc_user.append([np.round(num, 3) for num in acc_list_g])    
        test_acc_user_wo.append([np.round(num, 3) for num in acc_list_l])
        
        test_accuracy.append(np.round(np.mean(acc_list_g),3))
        test_accuracy_wo.append(np.round(np.mean(acc_list_l),3))
        
        print(f'\n | Test Accuracy with Global Training Round : {round + 1} |')
        print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
        print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
        print('For all users (with 2 type protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))
        
        
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, global_protos2)
    
    
    print("all rounds results")
    print("train_loss_avg=", train_loss)
    print("train_accuracy_avg=", train_accuracy)
    
    print("test_acc_user=", test_acc_user)
    print("test_acc_user_wo=", test_acc_user_wo)
    
    print("test_accuracy=", test_accuracy)
    print("test_accuracy_wo=", test_accuracy_wo)
    
    print("final round result")
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with 2 type protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)



if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.repeat(args.ways,args.num_users) # np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    # if args.dataset == 'mnist':
    #     k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    # elif args.dataset == 'cifar10':
    #     k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    if args.dataset =='cifar100':
        k_list = np.repeat(args.shots,args.num_users)
    # elif args.dataset == 'femnist':
    #     k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    #### dataset name
    elif args.dataset == 'CUB_200_2011':
        k_list = np.repeat(args.shots,args.num_users)
        
        # 이미지 
        for i in range(len(k_list)) :
            if k_list[i] > 30:
                k_list[i] = 30
 
    # class_list ##
    #all_class_list = np.random.choice(np.arange(0, 100), size=args.num_classes, replace=False)
    # print("class_list: ", class_list)

    ## class_list 추가
    #classes_list = [[0,1,2], [1,2,3],[2,3,4], [3,4,0], [0,1,4]]
    classes_list =args.classes_list
    # classes_list =[[0,1,3,8],[0,1,3,8], 
    #                [0,1,4,9],[0,1,4,9], 
    #                [0,1,5,10],[0,2,5,10],
    #                [0,2,5,10], [0,2,6,11],
    #                [0,2,6,11], [0,2,7,12],[0,2,7,12]]
    
    #train_dataset, test_dataset, user_groups, user_groups_lt, classes_list = get_dataset(args, n_list, k_list, all_class_list)
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list = get_dataset(args, n_list, k_list, classes_list)


    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        #### dataset name
        elif args.dataset == 'cifar100' or args.dataset == 'cifar10' or args.dataset == 'CUB_200_2011':
            if args.mode == 'model_heter':
                if i<10:
                    args.stride = [1,4]
                else:
                    args.stride = [2,2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)


        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)


    if args.mode == 'task_heter':
        # class_list 
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
        
        
        
start_time = time.time()

args = args_parser()
exp_details(args)










