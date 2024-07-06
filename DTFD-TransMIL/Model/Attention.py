import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from my_nystrom_attention import NystromAttention
from Model.network import Classifier_1fc

class Attention1(nn.Module):
    def __init__(self, dim=512,L=512,K=1):
        super(Attention1,self).__init__()
        self.L = L
        self.K = K
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
        self.dimreduction = nn.Linear(self.L, self.K)
        


    def forward(self, x):
        
      
        if x.dim() == 2:
           x = x[None]
        
        x = x + self.attn(self.norm(x))
        

        if x.dim() != 2:
           x = x.squeeze(0)
        x = self.dimreduction(x)
        x = torch.transpose(x, 1, 0)

        return x
    
class Attention2(nn.Module):
    def __init__(self, dim=512):
        super(Attention2,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
        
    def forward(self, x):
        
        
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class PPEG2(nn.Module):
    def __init__(self, dim=512):
        super(PPEG2, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = Attention2(dim=512)
        self.layer2 = Attention2(dim=512)
        self.norm = nn.LayerNorm(512)
        self.dimreduction = nn.Linear(512, 1)
  


    def forward(self, h,isNorm=True ):

        # h = kwargs['data'].float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        b, n, _ = h.shape

        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]

        #---->cls_token
        #B = h.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        #h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        h = h[:, -n:]
        

        h = self.norm(h)
        #h = self.dimreduction(h)
        
        h = h.squeeze(0)
                
        #---->cls_token
        #h = self.norm(h)[:,0]

        #h = torch.transpose(h, 1, 0)  # 1xN 

        #if isNorm:
            #h = F.softmax(h, dim=1)  # softmax over N
        
        return h
       
        #---->predict
        ##logits = self._fc2(h) #[B, n_classes]
        ##Y_hat = torch.argmax(logits, dim=1)
        ##Y_prob = F.softmax(logits, dim = 1)
        ##results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        ##return results_dict

class TransMIL2(nn.Module):
    def __init__(self):
        super(TransMIL2, self).__init__()
        self.pos_layer = PPEG2(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = Attention2(dim=512)
        self.layer2 = Attention2(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, 2)
        self.dimreduction = nn.Linear(512, 1)
  


    def forward(self, h,isNorm=True ):

        # h = kwargs['data'].float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        b, n, _ = h.shape

        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]
        

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        h = self._fc2(h) #[B, n_classes]
        
        return h
       

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN
       
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred