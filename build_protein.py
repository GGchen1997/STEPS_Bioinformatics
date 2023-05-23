import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import dgl
import dgl.data
from transformers import BertModel, BertTokenizer
import re
from dgl.nn import GINConv
import torch.backends.cudnn as cudnn
import higher 
import torch.optim as optim
import time
import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder
import random 

parser = PDBParser()
ppb=CaPPBuilder()

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def protein_preprocess(pdb_file='Q2G0W9'):
    #input: pdb_file
    #output(protein features):
    #   1. 1D residual seq
    #   2. distance matrix
    #   3. graph
    #   4. bond length
    #   5. angle

    pdb = "AF-" + pdb_file + "-F1-model_v3.pdb"
    #structure = parser.get_structure("Q2G0W9", "AF-Q2G0W9-F1-model_v1.pdb")
    structure = parser.get_structure(pdb_file, './pdbs/'+pdb)

    #1. 1D residual seq
    model = structure[0]
    pp = ppb.build_peptides(structure)
    seq = pp[0].get_sequence()
    seq = " ".join("".join(str(seq).split()))
    seq = re.sub(r"[UZOB]", "X", seq)

    #234
    Res = list(model.get_residues())
    N_Res = len(Res)
    distance_matrix  = np.zeros((N_Res, N_Res))
    point1 = []
    point2 = []
    bond_length = []
    for i in range(N_Res):
        for j in range(i+1, N_Res):
            #2. distance matrix
            distance_matrix[i, j] = Res[i]['CA'] - Res[j]['CA']
            distance_matrix[j, i] = distance_matrix[i, j]
            if distance_matrix[i, j] < 7:
               #3. graph
               point1 = point1 + [i, j]
               point2 = point2 + [j, i]
               #4. bond length
               bond_length = bond_length + [distance_matrix[i, j], distance_matrix[j, i]]
    graph = (point1, point2)
    #validate the sequence len and the struc len is the same or not
    #5. angle
    angle = np.zeros((N_Res, 2), dtype='float32')
    model.atom_to_internal_coordinates()
    for r in range(N_Res):
        res = Res[r]
        angle[r][0] = res.internal_coord.get_angle("phi")
        angle[r][1] = res.internal_coord.get_angle("psi")
    angle[0][0] = 0
    angle[N_Res-1][1] = 0
    tmp = "".join(seq.split())
    #print("len s", len(tmp), N_Res)
    if len(tmp) != N_Res:
        #print("error")
        #exit(0)
        return None
    return seq, distance_matrix.astype(np.float32), graph, np.array(bond_length).astype(np.float32), angle.astype(np.float32)

def load_pretrained():
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    pretrained_lm = BertModel.from_pretrained("Rostlab/prot_bert")
    freeze = True
    if freeze:
       modules = [pretrained_lm.embeddings, *pretrained_lm.encoder.layer[:29]]
       for module in modules:
           for param in module.parameters():
               param.requires_grad = False
    return tokenizer, pretrained_lm

class MGIN(nn.Module):
    def __init__(self, node_emb_size=1280):
        super(MGIN, self).__init__()
        self.node_emb_size = node_emb_size
        lin = torch.nn.Linear(node_emb_size, node_emb_size)
        self.gin = GINConv(lin, 'max')#.cuda()
        self.dis_nn = nn.Sequential(
                         nn.Linear(node_emb_size, 1),
                         nn.Sigmoid())
        self.mask_nn = nn.Sequential(
                          nn.Linear(node_emb_size, 2),
                          nn.Tanh())

    def forward(self, lm_embedding, node_feat, edge_feat, graph, mask_index=None):
        node_feat = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
        node_output = self.gin(graph, node_feat, edge_feat) + node_feat
        #distance pred
        dis_hid = node_output.view(1, -1, self.node_emb_size) - node_output.view(-1, 1, self.node_emb_size)
        dis_pred = self.dis_nn(dis_hid)
        #mask pred
        mask_pred = self.mask_nn(node_output[mask_index])
        return dis_pred, mask_pred

def my_model(tokenizer, pretrained_lm, mgin, seq, node_feat, edge_feat, graph, dis_mat, mask_index, mask_label):
    #lm embeddings
    seq = tokenizer(seq, return_tensors='pt')
    lm_embedding = pretrained_lm(**seq).last_hidden_state
    dis_pred, mask_pred = mgin(lm_embedding, node_feat, edge_feat, graph, mask_index)
    dis_loss = torch.mean(torch.clamp(torch.pow(dis_mat - dis_pred, 2), 0, 1))
    mask_loss = torch.mean(torch.pow(mask_pred - mask_label, 2))
    loss = dis_loss + mask_loss
    return loss

class Classifier(nn.Module):
    def __init__(self, cls_num=2, node_emb_size=1280):
        super(Classifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.pretrained_lm = BertModel.from_pretrained("Rostlab/prot_bert")
        lin = torch.nn.Linear(node_emb_size, node_emb_size)
        lin.weight.data = torch.load('Rostlab/gin.weight.pt')
        lin.bias.data = torch.load('Rostlab/gin.bias.pt')
        self.gin = GINConv(lin, 'max')
        self.head = nn.Sequential(
                         nn.Linear(node_emb_size, cls_num)
                         #nn.Softmax()
                         )
    def forward(self, seq, node_feat, edge_feat, graph):
        seq = self.tokenizer(seq, return_tensors='pt')
        lm_embedding = self.pretrained_lm(**seq).last_hidden_state
        node_feat = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
        node_output = self.gin(graph, node_feat, edge_feat) + node_feat
        graph_repre = torch.mean(node_output, dim=0)
        pred = self.head(graph_repre)
        return pred

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

def build_protein_data():
    #pdb_files = ['Q2G0W9']
    pdb_files = np.load("qualified_pdbs.npy", allow_pickle=True)
    pdb_files = list(pdb_files)
    pdb_data = {}
    i = 0
    for pdb_file in pdb_files:
        i = i + 1
        pdb_data[pdb_file] = protein_preprocess(pdb_file=pdb_file)
        np.save("./protein_struc/" + pdb_file + ".npy", pdb_data[pdb_file])
        print("finish {} protein {}".format(i, pdb_file))
        #break
    #np.save("protein.npy", pdb_data)

build_protein_data()
