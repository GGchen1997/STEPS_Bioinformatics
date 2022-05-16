import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import dgl
import dgl.data
from dgl.nn import GINConv
from dgl.nn import GraphConv
from transformers import BertModel, BertTokenizer, XLNetTokenizer, XLNetModel
import higher

import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder

import numpy as np
import re
import time
import random 

parser = PDBParser()
ppb=CaPPBuilder()

CE = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def adjust_learning_rate(optimizer, lr0, epoch, T):
    lr = lr0 * (1 + np.cos(np.pi * epoch * 1.0 / (T * 1.0))) / 2.0
    print("epoch {} use lr {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_pretrained(layer_num=29, model='bert'):
    if model in ['bert']:
       tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
       pretrained_lm = BertModel.from_pretrained("Rostlab/prot_bert")
       modules = [pretrained_lm.embeddings, *pretrained_lm.encoder.layer[:29]]
    elif model in ['xlnet']:
       tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
       xlnet_men_len = 512
       pretrained_lm = XLNetModel.from_pretrained("Rostlab/prot_xlnet",mem_len=xlnet_men_len)
       modules = [pretrained_lm.word_embedding, *pretrained_lm.layer[:29]]
    pretrained_lm = pretrained_lm.eval()
    print("pretrained_lm", pretrained_lm)
    freeze = True
    if freeze:
       for module in modules:
           for param in module.parameters():
               param.requires_grad = False
    return tokenizer, pretrained_lm

class MGIN(nn.Module):
    def __init__(self, use_lm=1, node_emb_size=1280, model='bert'):
        super(MGIN, self).__init__()
        self.node_emb_size = node_emb_size
        lin = torch.nn.Linear(node_emb_size, node_emb_size)
        self.gin = GINConv(lin, 'sum')#.cuda()
        lin1 = torch.nn.Linear(node_emb_size, node_emb_size)
        self.gin1 = GINConv(lin1, 'sum')
        self.dis_nn = nn.Sequential(
                         nn.Linear(node_emb_size, int(node_emb_size/10)),
                         nn.ReLU(inplace=True),
                         nn.Linear(int(node_emb_size/10), 30)
                         )
        self.mask_nn = nn.Sequential(
                         nn.Linear(node_emb_size, int(node_emb_size/10)),
                         nn.ReLU(inplace=True),
                         nn.Linear(int(node_emb_size/10), 2),
                         nn.Tanh()
                         )
        self.model = model
        self.use_lm = use_lm

    def obtain_embeds(self, seq, tokenizer, pretrained_lm, node_feat, edge_feat, graph, device):
        # lm embeddings
        seq = tokenizer(seq, return_tensors='pt')
        seq['attention_mask'] = seq['attention_mask'].to(device)
        seq['input_ids'] = seq['input_ids'].to(device)
        seq['token_type_ids'] = seq['token_type_ids'].to(device)
        if self.use_lm:
            lm_embedding = pretrained_lm(**seq).last_hidden_state
            if self.model in ['bert']:
                node_feat0 = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
                my_lm_embedding = lm_embedding.squeeze(0)[1:-1,:]
            elif self.model in ['xlnet']:
                node_feat0 = torch.cat([lm_embedding.squeeze(0)[:-2,:], node_feat], dim=1)
                my_lm_embedding = lm_embedding.squeeze(0)[:-2,:]
        else:
            lm_embedding = F.one_hot(seq['input_ids'][0, 1:-1], num_classes=1024)
            node_feat0 = torch.cat([lm_embedding, node_feat], dim=1)
        node_feat = self.gin(graph=graph, feat = node_feat0.data, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        node_output = self.gin1(graph=graph, feat = node_feat, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        #return 3D level information
        my_lm_embedding = torch.mean(my_lm_embedding, 0, True)
        node_output = torch.mean(node_output, 0, True)
        return my_lm_embedding.reshape(1, -1), node_output.reshape(1, -1)

    def forward(self, lm_embedding, node_feat, edge_feat, graph, mask_index=None):
        if self.use_lm:
            if self.model in ['bert']:
                node_feat0 = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
            elif self.model in ['xlnet']:
                node_feat0 = torch.cat([lm_embedding.squeeze(0)[:-2,:], node_feat], dim=1)
        else:
            node_feat0 = torch.cat([lm_embedding, node_feat], dim=1)
        node_feat = self.gin(graph=graph, feat = node_feat0.data, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        #1D level and 3D level added together
        node_output = self.gin1(graph=graph, feat = node_feat, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        if self.use_lm:
            node_output = node_output + node_feat0
        #distance pred
        dis_hid = torch.pow(node_output.view(1, -1, self.node_emb_size) - node_output.view(-1, 1, self.node_emb_size), 2)
        dis_pred = self.dis_nn(dis_hid.view(-1, self.node_emb_size))
        #mask pred
        mask_pred = self.mask_nn(node_output[mask_index])#*np.pi
        return dis_pred, mask_pred


def my_model(args, tokenizer, pretrained_lm, mgin, seq, node_feat, edge_feat, graph, dis_mat, mask_index, mask_label, device):
    #lm embeddings
    seq = tokenizer(seq, return_tensors='pt')
    seq['attention_mask'] = seq['attention_mask'].to(device)
    seq['input_ids'] = seq['input_ids'].to(device)
    seq['token_type_ids'] = seq['token_type_ids'].to(device)
    if args.use_lm:
        lm_embedding = pretrained_lm(**seq).last_hidden_state
    else:
        lm_embedding = F.one_hot(seq['input_ids'][0, 1:-1], num_classes=1024)
    dis_pred, mask_pred = mgin(lm_embedding, node_feat, edge_feat, graph, mask_index)
    dis_mat = torch.floor(torch.clamp(dis_mat.view(-1, 1), 0, 149)/5).long().squeeze()
    dis_loss = CE(dis_pred, dis_mat)
    mask_loss = torch.mean(torch.pow(mask_pred - mask_label, 2))
    return dis_loss, mask_loss

class ClassifierJoint(nn.Module):
    def __init__(self, cls_num=2, node_emb_size=1280, device=torch.device("cuda:0"), args=None):
        super(ClassifierJoint, self).__init__()

        self.use_lm = args.use_lm
        self.tokenizer, self.pretrained_lm = load_pretrained(model=args.base_model)
        self.pretrained_lm = self.pretrained_lm.to(device)
        self.pretrained_lm.eval()

        load_dir = '/scratch/canchen/mypretrained/' + str(args.use_lm) + "_" + str(args.prt_coeff) + "_" + args.prt_trade + "_" + args.base_model

        lin = torch.nn.Linear(node_emb_size, node_emb_size)
        lin.weight.data = torch.load(load_dir + "_joint_" + str(args.joint_coeff) + 'gin.weight.pt')
        lin.bias.data = torch.load(load_dir + "_joint_" + str(args.joint_coeff) + 'gin.bias.pt')
        self.gin = GINConv(lin, 'sum')

        lin1 = torch.nn.Linear(node_emb_size, node_emb_size)
        lin1.weight.data = torch.load(load_dir + "_joint_" + str(args.joint_coeff) + 'gin1.weight.pt')
        lin1.bias.data = torch.load(load_dir + "_joint_" + str(args.joint_coeff) + 'gin1.bias.pt')
        self.gin1 = GINConv(lin1, 'sum')

        self.head = nn.Sequential(
                         nn.Linear(node_emb_size, cls_num),
                         nn.Tanh()
                         )
        self.device = device
        self.model = args.base_model

    def forward(self, seq, node_feat, edge_feat, graph):
        with torch.no_grad():
           seq = self.tokenizer(seq, return_tensors='pt')
           seq['attention_mask'] = seq['attention_mask'].to(self.device)
           seq['input_ids'] = seq['input_ids'].to(self.device)
           seq['token_type_ids'] = seq['token_type_ids'].to(self.device)
           if self.use_lm:
              lm_embedding = self.pretrained_lm(**seq).last_hidden_state.data
              if self.model in ['bert']:
                 node_feat0 = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
              elif self.model in ['xlnet']:
                 node_feat0 = torch.cat([lm_embedding.squeeze(0)[:-2,:], node_feat], dim=1)
           else:
              lm_embedding = F.one_hot(seq['input_ids'][0, 1:-1], num_classes=1024)
              node_feat0 = torch.cat([lm_embedding, node_feat], dim=1)
        node_feat = self.gin(graph=graph, feat = node_feat0.data, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        node_output = self.gin1(graph=graph, feat = node_feat, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        if self.use_lm:
            node_output = node_output + node_feat0
        graph_repre = torch.mean(node_output, dim=0)
        pred = self.head(graph_repre)
        return pred

class Classifier(nn.Module):
    def __init__(self, cls_num=2, node_emb_size=1280, device=torch.device("cuda:0"), args=None):
        super(Classifier, self).__init__()

        self.use_lm = args.use_lm
        self.tokenizer, self.pretrained_lm = load_pretrained(model=args.base_model)
        self.pretrained_lm = self.pretrained_lm.to(device)
        self.pretrained_lm.eval()

        load_dir = '/scratch/canchen/mypretrained/' + str(args.use_lm) + "_" + str(args.prt_coeff) + "_" + args.prt_trade + "_" + args.base_model

        lin = torch.nn.Linear(node_emb_size, node_emb_size)
        lin.weight.data = torch.load(load_dir + "_" + 'gin.weight.pt')
        lin.bias.data = torch.load(load_dir + "_" + 'gin.bias.pt')
        self.gin = GINConv(lin, 'sum')

        lin1 = torch.nn.Linear(node_emb_size, node_emb_size)
        lin1.weight.data = torch.load(load_dir + "_" + 'gin1.weight.pt')
        lin1.bias.data = torch.load(load_dir + "_" + 'gin1.bias.pt')
        self.gin1 = GINConv(lin1, 'sum')

        self.head = nn.Sequential(
                         nn.Linear(node_emb_size, cls_num),
                         nn.Tanh()
                         )
        self.device = device
        self.model = args.base_model

    def forward(self, seq, node_feat, edge_feat, graph):
        with torch.no_grad():
           seq = self.tokenizer(seq, return_tensors='pt')
           seq['attention_mask'] = seq['attention_mask'].to(self.device)
           seq['input_ids'] = seq['input_ids'].to(self.device)
           seq['token_type_ids'] = seq['token_type_ids'].to(self.device)
           if self.use_lm:
              lm_embedding = self.pretrained_lm(**seq).last_hidden_state.data
              if self.model in ['bert']:
                 node_feat0 = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
              elif self.model in ['xlnet']:
                 node_feat0 = torch.cat([lm_embedding.squeeze(0)[:-2,:], node_feat], dim=1)
           else:
              lm_embedding = F.one_hot(seq['input_ids'][0, 1:-1], num_classes=1024)
              node_feat0 = torch.cat([lm_embedding, node_feat], dim=1)
        node_feat = self.gin(graph=graph, feat = node_feat0.data, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        node_output = self.gin1(graph=graph, feat = node_feat, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        if self.use_lm:
            node_output = node_output + node_feat0
        graph_repre = torch.mean(node_output, dim=0)
        pred = self.head(graph_repre)
        return pred

class DeepFRI(nn.Module):
    def __init__(self, cls_num=2, node_emb_size=1280, device=torch.device("cuda:0"), args=None):
        super(DeepFRI, self).__init__()

        self.use_lm = args.use_lm
        self.tokenizer, self.pretrained_lm = load_pretrained(model=args.base_model)
        self.pretrained_lm = self.pretrained_lm.to(device)
        self.pretrained_lm.eval()

        self.gcn = GraphConv(node_emb_size, node_emb_size, norm='both', weight=True, bias=True)
        self.gcn1 = GraphConv(node_emb_size, node_emb_size, norm='both', weight=True, bias=True)
        self.gcn2 = GraphConv(node_emb_size, node_emb_size, norm='both', weight=True, bias=True)

        self.head = nn.Sequential(
                         nn.ReLU(),
                         nn.Linear(node_emb_size*3, int(node_emb_size/10)),
                         nn.ReLU(),
                         nn.Linear(int(node_emb_size/10), cls_num),
                         nn.Tanh()
                         )
        self.device = device
        self.model = args.base_model

    def forward(self, seq, node_feat, edge_feat, graph):
        with torch.no_grad():
           seq = self.tokenizer(seq, return_tensors='pt')
           seq['attention_mask'] = seq['attention_mask'].to(self.device)
           seq['input_ids'] = seq['input_ids'].to(self.device)
           seq['token_type_ids'] = seq['token_type_ids'].to(self.device)
           lm_embedding = self.pretrained_lm(**seq).last_hidden_state.data
           if self.model in ['bert']:
              node_feat0 = torch.cat([lm_embedding.squeeze(0)[1:-1,:], torch.zeros_like(node_feat)], dim=1)
           elif self.model in ['xlnet']:
              node_feat0 = torch.cat([lm_embedding.squeeze(0)[:-2,:], torch.zeros_like(node_feat)], dim=1)
        node_feat1 = self.gcn(graph=graph, feat=node_feat0.data)
        node_feat2 = self.gcn1(graph=graph, feat=node_feat1)
        node_feat3 = self.gcn2(graph=graph, feat=node_feat2)
        node_output = torch.cat([node_feat1, node_feat2, node_feat3], dim=1)
        #node_output = torch.cat([node_feat0.data, node_feat1, node_feat2, node_feat3], dim=1)
        graph_repre = torch.mean(node_output, dim=0)
        pred = self.head(graph_repre)
        return pred

class BaseClassifier(nn.Module):
    def __init__(self, cls_num=2, node_emb_size=1024, device=torch.device("cuda:0"), args=None):
        super(BaseClassifier, self).__init__()
        self.tokenizer, self.pretrained_lm = load_pretrained(model=args.base_model)
        self.pretrained_lm.eval()
        self.head = nn.Sequential(
                         nn.Linear(node_emb_size, cls_num),
                         nn.Tanh()
                         )
        self.device = device
    def forward(self, seq0, node_feat, edge_feat, graph):
        with torch.no_grad():
          seq = self.tokenizer(seq0, return_tensors='pt')
          seq['attention_mask'] = seq['attention_mask'].to(self.device)
          seq['input_ids'] = seq['input_ids'].to(self.device)
          seq['token_type_ids'] = seq['token_type_ids'].to(self.device)
          lm_embedding = self.pretrained_lm(**seq).last_hidden_state.data
          graph_repre = torch.mean(lm_embedding.squeeze(0)[1:-1,:], dim=0)
        pred = self.head(graph_repre.data)
        return pred

class NodeEmb(nn.Module):
    def __init__(self, input_emb_size=1024, node_emb_size=1280):
        super(NodeEmb, self).__init__()
        self.L1 = nn.Linear(input_emb_size, node_emb_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.L2 = nn.Linear(node_emb_size, node_emb_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.node_emb_size = node_emb_size

    def forward(self, node_feat):
        if self.node_emb_size == node_feat.shape[1]:
            node_out = self.L1(node_feat) + node_feat
        else:
            node_out = self.L1(node_feat) + \
            torch.cat([node_feat, torch.zeros(node_feat.shape[0], self.node_emb_size-node_feat.shape[1]).to(device)], dim=1)
        node_out = self.relu1(node_out)
        node_out = self.L2(node_out) + node_out
        node_out = self.relu2(node_out)
        return node_out



class MutualInfo(nn.Module):
    def __init__(self, node_emb_size=1024):
        super(MutualInfo, self).__init__()
        self.seq_emb_layer = NodeEmb(input_emb_size=1024)
        self.struc_emb_layer = NodeEmb(input_emb_size=1280)
        self.softplus = nn.Softplus()
    def forward(self, seq_emb, stru_emb):
        seq_emb = self.seq_emb_layer(seq_emb)
        stru_emb = self.struc_emb_layer(stru_emb)
        distance = torch.mm(seq_emb, stru_emb.t())
        diag = torch.diag(distance)
        diag_loss = torch.mean(self.softplus(-diag))
        undiag = distance.flatten()[:-1].view(distance.shape[0] - 1, distance.shape[0] + 1)[:, 1:].flatten()
        undiag_loss = torch.mean(self.softplus(undiag))

        loss = diag_loss + undiag_loss
        return loss

def scalar2vec(angle, center):
    #angle seq_size*2
    #angle = torch.ones(4, 2)
    #center = torch.linspace(0, 2*np.pi, steps=3).view(1, -1)
    phi_angle = angle[:, 0].view(-1, 1)
    psi_angle = angle[:, 1].view(-1, 1)
    phi_vec = torch.exp(-10*torch.pow(phi_angle - center, 2))
    psi_vec = torch.exp(-10*torch.pow(psi_angle - center, 2))
    vec = torch.cat((phi_vec, psi_vec), dim=1)
    return vec

def remove_npy(array):
    new_array = []
    for a in array:
        tmp = a.split(".")[0]
        new_array.append(tmp)
    return new_array
