import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

import dgl
import higher

import time
import random
import argparse

from utils import *

#args
parser = argparse.ArgumentParser(description='Bilevel Protein Pretraining')
parser.add_argument('--mode', choices=['prt', 'ft'], type=str, default='prt')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--interval', default=100, type=int)
parser.add_argument("--base_model", choices=['bert', 'xlnet'], type=str, default='bert')
#pretrain
parser.add_argument("--prt_lr", default=1e-3, type=float)
parser.add_argument('--prt_epochs', default=5, type=int)
parser.add_argument("--prt_wd", default=0.0, type=float)
parser.add_argument("--mask_ratio", default=0.15, type=float)
parser.add_argument("--prt_coeff", default=0.1, type=float)
parser.add_argument("--prt_trade", choices=['both', 'global', 'local'], default='both', type=str)
parser.add_argument('--use_lm', default=1, type=int)
#finetune
parser.add_argument('--task', choices=['loc', 'water', 'enzyme'], type=str, default='loc')
parser.add_argument('--ft_mode', choices=['base', 'bilevel-h', 'bilevel-b', 'deepfri'], type=str, default='bilevel-b')
parser.add_argument("--ft_lr", default=1e-4, type=float)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument("--ft_wd", default=0.0, type=float)
#parse
args = parser.parse_args()

#global params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
center = torch.linspace(-np.pi, np.pi, steps=128).view(1, -1).to(device)

def pretrain(args):
    #data
    train_index1 = list(np.load("deeploc/stru_pdb.npy", allow_pickle=True))
    print("train index1 len", len(train_index1))
    train_index2 = list(np.load("deeploc/enzyme_stru.npy", allow_pickle=True))
    print("train index2 len", len(train_index2))
    train_index = train_index1 + train_index2
    print("train index len", len(train_index))
    seq_embs = []
    struc_embs = []
    previous_seq_emb = None
    previous_struc_emb = None
    #model def
    tokenizer, pretrained_lm = load_pretrained(model=args.base_model)
    pretrained_lm = pretrained_lm.to(device)
    mgin = MGIN(use_lm=args.use_lm)
    mgin = mgin.to(device)
    mutualinfo = MutualInfo()
    mutualinfo = mutualinfo.to(device)
    mutualinfo.load_state_dict(torch.load("mutualinfo.pt"))
    #interpret prt trade
    if args.prt_trade == 'both':
        global_trade = 1.0
        local_trade = 1.0
    elif args.prt_trade == 'global':
        global_trade = 1.0
        local_trade = 0.0
    elif args.prt_trade == 'local':
        global_trade = 0.0
        local_trade = 1.0
    #optimizer def
    pretrain_inner_opt = optim.SGD(pretrained_lm.parameters(), lr=args.prt_lr*args.prt_coeff, weight_decay=args.prt_wd)
    gnn_opt = optim.Adam(mgin.parameters(), lr=args.prt_lr, weight_decay=args.prt_wd)
    mutualinfo_opt = optim.Adam(mutualinfo.parameters(), lr=0.0001, weight_decay=args.prt_wd)
    #training
    for e in range(args.prt_epochs):
        print('current training epoch is {}'.format(e))
        random.shuffle(train_index)
        #batch info
        batch_dis_loss = 0
        batch_mask_loss = 0
        idx = 0
        #change lr
        adjust_learning_rate(pretrain_inner_opt, args.prt_lr*args.prt_coeff, e, args.prt_epochs)
        adjust_learning_rate(gnn_opt, args.prt_lr, e, args.prt_epochs)
        adjust_learning_rate(mutualinfo_opt, 0.1, e, args.prt_epochs)
        start = time.time()
        for index in train_index:
            #feature extraction
            seq, distance_matrix, graph, bond_length, angle = np.load('./all_protein_struc/' + index + '.npy', allow_pickle=True)
            graph = dgl.graph(graph).to(device)
            angle[np.isnan(angle)] = 0.0
            scalar_angle = (torch.tensor(angle)/180).to(device)
            angle = scalar2vec(scalar_angle, center)
            bond_length = torch.tensor(bond_length).to(device)
            distance_matrix = torch.tensor(distance_matrix).to(device)
            #mask generation
            L = angle.shape[0]
            mask_index = torch.randperm(L)[0:max(int(args.mask_ratio*L), 2)]
            mask_label = scalar_angle[mask_index]
            angle[mask_index] = torch.zeros(angle[mask_index].shape).to(device)
            #optimization
            with higher.innerloop_ctx(pretrained_lm, pretrain_inner_opt) as (fmodel, diffopt):
                if args.use_lm:
                    #inner loop
                    with torch.backends.cudnn.flags(enabled=False):
                        seq_emb, struc_emb = mgin.obtain_embeds(seq, tokenizer, fmodel, angle, bond_length, graph, device)
                        if None != previous_seq_emb:
                            seq_emb_mutual = torch.cat([seq_emb, previous_seq_emb], dim=0)
                            struc_emb_mutual = torch.cat([struc_emb, previous_struc_emb], dim=0)
                        else:
                            seq_emb_mutual = seq_emb
                            struc_emb_mutual = struc_emb
                        loss_in = mutualinfo(seq_emb_mutual, struc_emb_mutual)
                        seq_embs.append(seq_emb.data)
                        struc_embs.append(struc_emb.data)
                        previous_seq_emb = seq_emb.data
                        previous_struc_emb = struc_emb.data
                    fmodel.zero_grad()
                    diffopt.step(loss_in)
                dis_loss_out, mask_loss_out = my_model(args, tokenizer, fmodel, mgin, seq, angle, bond_length, graph, distance_matrix, mask_index, mask_label, device)
                loss_out = global_trade*dis_loss_out + local_trade*mask_loss_out
                gnn_opt.zero_grad()
                loss_out.backward()
                gnn_opt.step()
            #record
            idx = idx + 1
            batch_dis_loss = batch_dis_loss + global_trade*dis_loss_out.data
            batch_mask_loss = batch_mask_loss + local_trade*mask_loss_out.data
            if ((idx%2)==0) and idx and args.use_lm:
                tmp = torch.cat(seq_embs, dim=0)
                mutualinfo_loss = mutualinfo(torch.cat(seq_embs, dim=0), torch.cat(struc_embs, dim=0))
                mutualinfo_opt.zero_grad()
                mutualinfo_loss.backward()
                mutualinfo_opt.step()
                seq_embs = []
                struc_embs = []
            if ((idx % args.interval) == 0) and (idx != 0):
                print("avg dis loss is {} avg mask loss {}".format(batch_dis_loss/args.interval, batch_mask_loss/args.interval))
                print("time cost", time.time() - start)
                start = time.time()
                batch_dis_loss = 0
                batch_mask_loss = 0
            #break
        #break
    print("begin saving")
    #use_lm for table1-3; prt_coeff for table4; prt_trade for table5; base model for table6;
    save_prefix = '/scratch/canchen/mypretrained/' + str(args.use_lm) + "_" + str(args.prt_coeff) + "_" + args.prt_trade + "_" + args.base_model
    torch.save(mgin.gin.apply_func.weight.cpu().data, save_prefix + "_" + 'gin.weight.pt')
    torch.save(mgin.gin.apply_func.bias.cpu().data, save_prefix +  "_" + 'gin.bias.pt')
    torch.save(mgin.gin1.apply_func.weight.cpu().data, save_prefix + "_" + 'gin1.weight.pt')
    torch.save(mgin.gin1.apply_func.bias.cpu().data, save_prefix + "_" + 'gin1.bias.pt')
    print("finish saving")


def finetune(args):
    #data
    if args.task in ['loc', 'water']:
        stru_pdb = list(np.load('./deeploc/stru_pdb.npy', allow_pickle=True))
        train_data = np.load('./deeploc/'+ args.task + '/train.npy', allow_pickle=True).item()
        train_index = list(train_data.keys())
        train_index = list(set(train_index) & set(stru_pdb))
        test_data = np.load('./deeploc/'+ args.task + '/test.npy', allow_pickle=True).item()
        test_index = list(test_data.keys())
        test_index = list(set(test_index) & set(stru_pdb))
    elif args.task in ['enzyme']:
        stru_pdb = list(np.load('./deeploc/enzyme_stru.npy', allow_pickle=True))
        train_data = np.load('./enzyme/'+ args.task + '/train.npy', allow_pickle=True).item()
        train_index = remove_npy(list(train_data.keys()))
        train_index = list(set(train_index) & set(stru_pdb))
        test_data = np.load('./enzyme/'+ args.task + '/test.npy', allow_pickle=True).item()
        test_index = remove_npy(list(test_data.keys()))
        test_index = list(set(test_index) & set(stru_pdb))
    #classifier
    if args.task == 'loc':
        cls_num = 10
    elif args.task == 'water':
        cls_num = 2
    elif args.task == 'enzyme':
        cls_num = 384

    if args.ft_mode == 'base':
        model = BaseClassifier(cls_num = cls_num, device=device, args=args).to(device)
        cls_opt = optim.Adam(model.head.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)
    elif args.ft_mode in ['bilevel-h', 'bilevel-b']:
        model = Classifier(cls_num = cls_num, device=device, args=args).to(device)
        if args.ft_mode == 'bilevel-h':
            cls_opt = optim.Adam(model.head.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)
        elif args.ft_mode == 'bilevel-b':
            cls_opt = optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)
    elif args.ft_mode in ['deepfri']:
        model = DeepFRI(cls_num = cls_num, device=device, args=args).to(device)
        cls_opt = optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)
    #define loss function
    CE = nn.CrossEntropyLoss()
    for e in range(args.ft_epochs):
        print('current training epoch is {}'.format(e))
        random.shuffle(train_index)
        #batch info
        batch_loss = 0
        idx = 0
        #lr change
        adjust_learning_rate(cls_opt, args.ft_lr, e, args.ft_epochs)
        for index in train_index:
            #feature extraction
            seq, distance_matrix, graph, bond_length, angle = np.load('./all_protein_struc/' + index + '.npy', allow_pickle=True)
            #protein_data['Q2G0W9']
            graph = dgl.graph(graph).to(device)
            angle[np.isnan(angle)] = 0.0
            scalar_angle = (torch.tensor(angle)/180).to(device)
            angle = scalar2vec(scalar_angle, center)
            bond_length = torch.tensor(bond_length).to(device)
            #label pred
            label_pred = model(seq, angle, bond_length, graph).view(1, -1)
            #compute loss
            if args.task == 'enzyme':
                index = index + ".npy"
            loss = CE(label_pred, torch.LongTensor([train_data[index][1]]).to(device))
            cls_opt.zero_grad()
            loss.backward()
            cls_opt.step()
            #record
            idx = idx + 1
            batch_loss = batch_loss + loss.data
            if ((idx % args.interval) == 0) and (idx!=0):
                print("avg_loss is {}".format(batch_loss/args.interval))
                batch_loss = 0
            #break
        print('current test epoch is {}'.format(e))
        right_count = 0
        for index in test_index:
            #feature extraction
            seq, distance_matrix, graph, bond_length, angle = np.load('./all_protein_struc/' + index + '.npy', allow_pickle=True)
            #seq, distance_matrix, graph, bond_length, angle = protein_data['Q2G0W9']
            graph = dgl.graph(graph).to(device)
            scalar_angle = (torch.tensor(angle)/180).to(device)
            angle = scalar2vec(scalar_angle, center)
            bond_length = torch.tensor(bond_length).to(device)
            #label pred
            label_pred = model(seq, angle, bond_length, graph)
            #compute loss
            if args.task == 'enzyme':
                index = index + ".npy"
            if torch.LongTensor([test_data[index][1]]).to(device) == label_pred.argmax():
                right_count = right_count + 1
            #break
        print("test acc is {}".format(right_count/len(test_index)))
    print("final test acc is {}".format(right_count/len(test_index)))

        #break
     

if __name__ == '__main__':
    #print
    print(args)
    #set seed
    set_seed(args.seed)
    #training
    if args.mode == 'prt':
       pretrain(args)
    elif args.mode == 'ft':
       finetune(args)
