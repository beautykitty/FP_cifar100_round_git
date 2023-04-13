#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from models import CNNFemnist
from utils import compute_class_accuracy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        ## label + 1
        return torch.tensor(image), torch.tensor(label) #torch.tensor(label+1)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        
        #print("idxs:", idxs, "\n")
        idxs_train = idxs[:int(1 * len(idxs))]
        #print("idxs_train: ", idxs_train, "\n")
        # shuffle = False
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        
        #a, b = next(iter(trainloader))
        #print(b)
        
        return trainloader

    def update_weights(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_prox(self, idx, local_weights, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    ###
    def update_weights_het(self, args, idx, global_protos, global_protos2, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            ###
            agg_protos_label = {}
            agg_protos2_label = {}
            
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                #print("labels:", labels)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                ###
                log_probs, protos, protos2 = model(images)
                
                
                # print("log_probs: ", log_probs.shape)
                # print("protos shape: ", protos.shape)
                # print("protos2 shape: ", protos2.shape)

                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    ###
                    loss2 = 0*loss1
                    loss3 = 0*loss1
                    
                else:
                    ###
                    proto_new = copy.deepcopy(protos.data)
                    proto2_new = copy.deepcopy(protos2.data)
                    
                    
                    
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            ###
                            proto_new[i, :] = global_protos[label.item()][0].data
                            proto2_new[i,:] = global_protos2[label.item()][0].data
                        i += 1
                    ###
                    loss2 = loss_mse(proto_new, protos)
                    loss3 = loss_mse(proto2_new, protos2)
                
                ### 
                loss = loss1 + loss2 * args.ld + loss3 * args.ld2
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        ###
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                        agg_protos2_label[label_g[i].item()].append(protos2[i,:])
                        
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]
                        agg_protos2_label[label_g[i].item()] = [protos2[i,:]]
                        

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                ### 
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['3'].append(loss3.item())
            
            ###
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['3'].append(sum(batch_loss['3']) / len(batch_loss['3']))
        
        ### 
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['3'] = sum(epoch_loss['3']) / len(epoch_loss['3'])
        
        ###
        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label, agg_protos2_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


def test_inference(args, model, test_dataset, global_protos):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc

def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(100):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc

def test_inference_new_het_wo2(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], global_protos2=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_l = []
        
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=args.test_shots*args.ways, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos, protos2 = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            
            # print("pred_labels: ", pred_labels)
            # print("labels: ", labels)
            # pred_labels_user_wo.append(pred_labels)
            # labels_user_wo.append(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)
        #
#         print("pred_labels_user_wo=", pred_labels)
#         print("labels_user_wo=", labels)
           
    return acc_list_l


def test_inference_new_het_w2(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], global_protos2=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    loss_list = [] 
    
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=args.test_shots*args.ways, shuffle=True)

        # test (local model)
        model.eval()
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                ###
                outputs, protos, protos2 = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 1000
                ###
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                dist2 = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device) 
                
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            ###
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            d2 = loss_mse(protos2[i,:], global_protos2[j][0])
                            dist[i, j] = d
                            dist2[i,j] = d2

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                
                
                # pred_labels_user.append(pred_labels)
                # labels_user.append(labels)


                # compute loss
                ###
                proto_new = copy.deepcopy(protos.data)
                proto2_new = copy.deepcopy(protos2.data)
                
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                ###
                loss2 = loss_mse(proto_new, protos)
                loss3 = loss_mse(proto2_new, protos2)
                
                ###
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                    loss3 = loss3.cpu().detach().numpy()

                else:
                    loss2 = loss2.detach().numpy()
                    loss3 = loss3.detach().numpy()


            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            
#             print("pred_labels_user=", pred_labels)
#             print("labels=", labels)

            loss_list.append(loss2)
            loss_list.append(loss3)
    
    return acc_list_g

def test_inference_new_het_wo(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], global_protos2=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_l = []
        
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=args.test_shots*args.ways, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos, protos2 = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            
            # print("pred_labels: ", pred_labels)
            # print("labels: ", labels)
            # pred_labels_user_wo.append(pred_labels)
            # labels_user_wo.append(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)
        #
        print("pred_labels_user_wo=", pred_labels)
        print("labels_user_wo=", labels)
           
    return acc_list_l


def test_inference_new_het_w(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], global_protos2=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    loss_list = [] 
    
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=args.test_shots*args.ways, shuffle=True)

        # test (local model)
        model.eval()
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                ###
                outputs, protos, protos2 = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 1000
                ###
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                dist2 = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device) 
                
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            ###
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            d2 = loss_mse(protos2[i,:], global_protos2[j][0])
                            dist[i, j] = d
                            dist2[i,j] = d2

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                
                
                # pred_labels_user.append(pred_labels)
                # labels_user.append(labels)


                # compute loss
                ###
                proto_new = copy.deepcopy(protos.data)
                proto2_new = copy.deepcopy(protos2.data)
                
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                ###
                loss2 = loss_mse(proto_new, protos)
                loss3 = loss_mse(proto2_new, protos2)
                
                ###
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                    loss3 = loss3.cpu().detach().numpy()

                else:
                    loss2 = loss2.detach().numpy()
                    loss3 = loss3.detach().numpy()


            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            
            print("pred_labels_user=", pred_labels)
            print("labels=", labels)

            loss_list.append(loss2)
            loss_list.append(loss3)
    
    return acc_list_g



###
def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], global_protos2=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    
    pred_labels_user_wo = []
    labels_user_wo = []
    
    pred_labels_user = []
    labels_user = [] 
    
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=args.test_shots*args.ways, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos, protos2 = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            
            # print("pred_labels: ", pred_labels)
            # print("labels: ", labels)
            # pred_labels_user_wo.append(pred_labels)
            # labels_user_wo.append(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)
        #
        print("pred_labels_user_wo=", pred_labels)
        print("labels_user_wo=", labels)
        

        # test (use global proto)
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                ###
                outputs, protos, protos2 = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 1000
                ###
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                dist2 = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device) 
                
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            ###
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            d2 = loss_mse(protos2[i,:], global_protos2[j][0])
                            dist[i, j] = d
                            dist2[i,j] = d2

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                
                
                # pred_labels_user.append(pred_labels)
                # labels_user.append(labels)


                # compute loss
                ###
                proto_new = copy.deepcopy(protos.data)
                proto2_new = copy.deepcopy(protos2.data)
                
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                ###
                loss2 = loss_mse(proto_new, protos)
                loss3 = loss_mse(proto2_new, protos2)
                
                ###
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                    loss3 = loss3.cpu().detach().numpy()

                else:
                    loss2 = loss2.detach().numpy()
                    loss3 = loss3.detach().numpy()


            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            
            print("pred_labels_user=", pred_labels)
            print("labels=", labels)

            
            ###
            loss_list.append(loss2)
            loss_list.append(loss3)
            
    

    
    
    return acc_list_l, acc_list_g, loss_list

def test_inference_new_het_lt2(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], global_protos2=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    
    pred_labels_user_wo = []
    labels_user_wo = []
    
    pred_labels_user = []
    labels_user = [] 
    
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=args.test_shots*args.ways, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos, protos2 = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            
            # print("pred_labels: ", pred_labels)
            # print("labels: ", labels)
            # pred_labels_user_wo.append(pred_labels)
            # labels_user_wo.append(labels)

        acc = correct / total
        #print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)
        #
        #print("pred_labels_user_wo=", pred_labels)
        #print("labels_user_wo=", labels)
        

        # test (use global proto)
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                ###
                outputs, protos, protos2 = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 1000
                ###
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                dist2 = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device) 
                
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            ###
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            d2 = loss_mse(protos2[i,:], global_protos2[j][0])
                            dist[i, j] = d
                            dist2[i,j] = d2

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                
                
                # pred_labels_user.append(pred_labels)
                # labels_user.append(labels)


                # compute loss
                ###
                proto_new = copy.deepcopy(protos.data)
                proto2_new = copy.deepcopy(protos2.data)
                
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                ###
                loss2 = loss_mse(proto_new, protos)
                loss3 = loss_mse(proto2_new, protos2)
                
                ###
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                    loss3 = loss3.cpu().detach().numpy()

                else:
                    loss2 = loss2.detach().numpy()
                    loss3 = loss3.detach().numpy()


            acc = correct / total
            #print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            
            #print("pred_labels_user=", pred_labels)
            #print("labels=", labels)

            
            ###
            loss_list.append(loss2)
            loss_list.append(loss3)
            
    

    
    
    return acc_list_l, acc_list_g, loss_list

def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt+=1
        if cnt==20:
            break

    acc = correct/total

    return acc



if __name__ == '__main__':

    from tqdm import tqdm
    from pathlib import Path
    import copy, sys
    import time
    import random
    import torch.utils.model_zoo as model_zoo


    from options import args_parser    

    lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))
    mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    
    
    from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
    from resnet import resnet18
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
    
    args = args_parser()
    
    
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
    
    
    
    n_list = np.repeat(args.ways,args.num_users)
    k_list = np.repeat(args.shots,args.num_users)
    
    # for i in range(len(k_list)) :
    #     if k_list[i] > 30:
    #         k_list[i] = 30
            
    
    all_class_list = np.random.choice(np.arange(0, 100), size=args.num_classes, replace=False)
    #all_class_list = np.random.choice(np.arange(1, 201), size=args.num_classes, replace=False)


    ## class_list 추가
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list = get_dataset(args, n_list, k_list)
    
    local_model_list = []
    for i in range(args.num_users):

        if args.dataset == 'CUB_200_2011' or 'cifar100':
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
    
    global_protos = []
    global_protos2 = []
    idxs_users = np.arange(args.num_users)
    
    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos, local_protos2 = [], [], {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        ###
        proto_loss = 0
        proto2_loss = 0
        
        for idx in idxs_users:
            print("user: ", idx)
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
            
            # summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            # summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            # summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            # summary_wSriter.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            
            ###
            proto_loss += loss['2']
            proto2_loss += loss['3']
            
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

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos, global_protos2)




