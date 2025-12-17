#%%

import math
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

import torchaudio
from torchaudio.models import wav2vec2_base, wav2vec2_large, wav2vec2_large_lv60k

from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers import WavLMConfig, WavLMModel
from transformers import HubertConfig, HubertModel

import time
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)

            
def xavier_init(*modules):
    for m in modules:
        torch.nn.init.xavier_uniform_(m)
        
def kaiming_init(*modules):
    for m in modules:
        torch.nn.init.kaiming_uniform_(m, nonlinearity='relu')
        
def zero_init(*modules):
    for m in modules:
        torch.nn.init.zeros_(m)

def apply_xavier_init(m:nn.Module):
    for name, p in m.named_parameters():
        if 'weight' in name: xavier_init(p)
        elif 'bias' in name: zero_init(p)

def apply_kaiming_init(m:nn.Module):
    for name, p in m.named_parameters():
        if 'weight' in name: kaiming_init(p)
        elif 'bias' in name: zero_init(p)

def init_weights(m:nn.Module):
    for name, p in m.named_parameters():
        if 'weight' in name and len(p.size()) > 1: 
            torch.nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            torch.nn.init.zeros_(p)
                

#%%


class ConvReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same')
        self.act  = nn.ReLU()
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """ x: (B C H W) """
        return self.bn(self.act(self.conv(x)))


class SE_Module(nn.Module):
    """ squeeze and excitation networks, 2018 """
    def __init__(self, channels, se_channels=128, p_dropout=0.1):
        super().__init__()
        # se_channels = int(channels*reduction)
        
        self.se_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(channels, se_channels, kernel_size=(1,1)),
            nn.ReLU(),
            # nn.BatchNorm2d(se_channels),
            nn.Conv2d(se_channels, channels, kernel_size=(1,1)),
            nn.Sigmoid(),
            nn.Dropout(p=p_dropout) if p_dropout > 0 else nn.Identity()
        )
        
    def forward(self, x):
        return self.se_attention(x) * x


class Res2ConvReluBn(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, scale=8):
        super().__init__()
        assert channels % scale == 0

        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([
            ConvReluBn(self.width, self.width, kernel_size, dilation) for _ in range(self.nums)
        ])

    def forward(self, x):
        """ x """

        output = []
        spx = torch.split(x, self.width, dim=1)
        sp  = spx[0]

        for i, layer in enumerate(self.convs):
            if i >= 1:
                sp = sp + spx[i]
            
            sp = layer(sp)
            output.append(sp)

        if self.scale != 1:
            output.append(spx[self.nums])
        
        output = torch.cat(output, dim=1)

        return output


class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()

        self.se_res2block = nn.Sequential(
            ConvReluBn(channels, channels, kernel_size=1),
            Res2ConvReluBn(channels, kernel_size, dilation, scale),
            ConvReluBn(channels, channels, kernel_size=1),
            SE_Module(channels, p_dropout=0)
        )
    
    def forward(self, x):
        return x + self.se_res2block(x)


class ECAPA(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            ConvReluBn(in_channels, channels, kernel_size=(5,3))
        )
        self.block1  = SE_Res2Block(channels, kernel_size=(3,3), stride=1, padding=2, dilation=2, scale=8)
        self.block2  = SE_Res2Block(channels, kernel_size=(3,3), stride=1, padding=3, dilation=3, scale=8)
        self.block3  = SE_Res2Block(channels, kernel_size=(3,3), stride=1, padding=4, dilation=4, scale=8)

        self.dense_channels = channels * 3
        self.fusion = nn.Sequential(
            nn.Conv2d(self.dense_channels, self.dense_channels, kernel_size=(1,1)),
            nn.ReLU()
        )
    
    def get_out_dim(self):
        return self.dense_channels
    
    def forward(self, x):
        """ x: (B C H=T W=L) """

        x = self.in_conv(x)

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        output = self.fusion(torch.cat([x1, x2, x3], dim=1))

        return output



class LayerAttentionPool(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        d_k = out_dim // n_heads
        
        self.in_projection = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, (self.n_heads * d_k), kernel_size=(1, 1), bias=False)
        ); apply_xavier_init(self.in_projection[-1])
        
        self.W1 = nn.Parameter(torch.randn(self.n_heads, n_layers, n_layers//2))
        self.b1 = nn.Parameter(torch.zeros(n_heads, 1, n_layers//2))
        kaiming_init(self.W1); zero_init(self.b1)

        self.W2 = nn.Parameter(torch.randn(self.n_heads, n_layers//2, n_layers))
        self.b2 = nn.Parameter(torch.zeros(n_heads, 1, n_layers))
        xavier_init(self.W2), zero_init(self.b2)

        self.out_projection = nn.Sequential(
            nn.Conv1d((self.n_heads * d_k), out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim)
        ); apply_xavier_init(self.out_projection[0])
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        """ 
        """
        B = x.size(0)
        
        x = self.in_projection(x)
        x = rearrange(x, 'B (h k) L T -> (B T) (h L) k', h=self.n_heads) # ([B x T], [h x C], k)
        
        avg_pool = rearrange(F.adaptive_avg_pool1d(x, 1), "BT (h L) 1 -> BT h 1 L", h=self.n_heads)
        max_pool = rearrange(F.adaptive_max_pool1d(x, 1), "BT (h L) 1 -> BT h 1 L", h=self.n_heads) # ([B x T], h, 1, L)

        avg_pool_bnk = torch.matmul(F.relu((torch.matmul(avg_pool, self.W1) + self.b1)), self.W2) + self.b2
        max_pool_bnk = torch.matmul(F.relu((torch.matmul(max_pool, self.W1) + self.b1)), self.W2) + self.b2 # ([B x T], h, 1, C)
        
        pool_sum = avg_pool_bnk + max_pool_bnk
        attn_scr = self.dropout( rearrange(pool_sum.sigmoid(), "BT h 1 L -> BT (h L) 1") ) # ([B x T], [h x L], 1)
        x = rearrange(x * attn_scr, 'BT (h L) k -> BT (h k) L', h=self.n_heads) # ([B x T], [h x L], k) -> ([B x T], [h x k], L)
        
        output = F.adaptive_max_pool1d(x, 1) # ([B * T], [h * k], 1)
        output = rearrange(output, '(B T) hk 1 -> B hk T', B=B)
        output = self.out_projection(output) # (B D T)
        
        # attn_scr = rearrange(attn_scr, '(B T) (h L) 1 -> B h L T', B=B, h=self.n_heads)
        
        return output


class AttentiveStatisticPool(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim * 3, in_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim // 2),
            # nn.Tanh(), # I add this layer
            nn.Conv1d(in_dim // 2, in_dim, kernel_size=1),
            )
        self.dropout = nn.Dropout(p=dropout)
        self.attn_norm = nn.BatchNorm1d(in_dim * 2)
        
        self.out_proj = nn.Linear(in_dim * 2, out_dim)
        self.out_norm = nn.BatchNorm1d(out_dim)
        
    def forward(self, x:torch.Tensor, mask=None):
        """ 
        Input:
            x: (B C1 T) - float
            mask: (B T) - bool
        Output:
            x: (B C2)
        """
        T = x.size(-1)
        
        if mask is not None:
            mask = mask[:, None, :] # (B 1 T)
            N    = mask.sum(dim=-1, keepdim=True) # (B 1 1)
            mu = (x * mask).sum(dim=-1, keepdim=True) / N # (B C1 1)
            sg = torch.sqrt((((x - mu) ** 2) * mask).sum(dim=-1, keepdim=True) / N) # (B C1 1)
        else:
            mu = x.mean(dim=-1, keepdim=True) # (B C1 1)
            sg = x.std(dim=-1, keepdim=True)  # (B C1 1)
            
        stat_pool = torch.cat([x, mu.expand(-1,-1,T), sg.expand(-1,-1,T)], dim=1) # (B 3C T)
        
        attn_scr = self.attention(stat_pool) # (B C1 T)
        if mask is not None:
            attn_scr.masked_fill_(~mask, torch.finfo(torch.float32).min) # mask: (B 1 T)
        attn_scr = self.dropout(F.softmax(attn_scr, dim=-1)) # (B C1 T)
        
        attn_mu = torch.sum(x * attn_scr, dim=-1) # (B C1)
        attn_sg = torch.sqrt((torch.sum((x**2) * attn_scr, dim=-1) - attn_mu**2).clamp(min=1e-4)) # (B C1)
        attn_pool = torch.cat([attn_mu, attn_sg], dim=1) # (B 2xC1)
        attn_pool = self.attn_norm(attn_pool)
        
        x = self.out_norm(self.out_proj(attn_pool))
        return x


class Multiheaded_AAMsoftmax(nn.Module):
    def __init__(self, in_dim:int, n_heads:int, n_class:int, m:float, s:float, d_k=None):
        super().__init__()
        
        # local-variables
        self.m = m
        self.s = s        
        self.n_heads = n_heads
        self.n_class = n_class
        if d_k is None: 
            assert in_dim % self.n_heads == 0
            d_k = in_dim // n_heads
        
        # setup for angular additive marginal softmax
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        
        self.weight = nn.Parameter(torch.rand(self.n_heads, d_k, self.n_class))
        xavier_init(self.weight)
        
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, x, label):
        """
        Input:
            x: (B, D)
            label: (B,)
        Output:
            output: (B, C)
            loss: (1,)
        """
        # head-wise split & inter-vector cosine/sine -> angular measure
        x = rearrange(x, 'B (h k) -> B h 1 k', h=self.n_heads)
        cosine = torch.matmul(F.normalize(x, dim=-1),
                              F.normalize(self.weight, dim=1)).squeeze(2) # (B, h, C)
        sine = torch.sqrt( (1.0 - torch.mul(cosine, cosine)).clamp(0, 1) )
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        # one-hot label for each heads
        label = label[:, None].expand(label.size(0), self.n_heads) # (B, h)
        one_hot = torch.zeros_like(cosine) # (B, h, C)
        one_hot.scatter_(-1, label[..., None], 1) # scatter value-1 on c-th class
        
        # calculate loss with batchfying the heads
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # (B, h, C)
        output = output * self.s
        loss = self.ce(output.reshape(-1, self.n_class), label.reshape(-1)) # ([B x h], C)
        
        # get predicted class probability
        with torch.no_grad():
            output = F.softmax(output, dim=-1).mean(dim=1) # (B, h, C).mean(dim=h) -> (B, C)
            
        return output, loss


#%%

class SVmodel(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        
        # [HYPERPARAMS] _______________________________________________________________________________________________________________________
        # backbone
        self.BACKBONE_CFG = config['model']['backbone_cfg']
        self.USE_PRETRAIN = config['model']['use_pretrain']
        self.FRZ_PRETRAIN = config['model']['frz_pretrain']
        
        # net
        self.CONV_DIM = config['model']['conv_dim']
        # self.N_BLOCKS = config['model']['n_blocks']
        # self.DEPTH    = config['model']['depth']

        # LayerPool
        self.HIDDEN_DIM = config['model']['hidden_size']
        self.N_LAYERS = config['model']['n_layer']
        self.N_HEADS  = config['model']['n_head']

        # AttnPool
        self.POOL_DIM = config['model']['pool_dim']

        # Loss
        self.NUM_TRAIN_CLASS = config['data']['nb_class_train']
        self.NUM_LOSS_HEADS  = config['model']['num_loss_heads']

        # [SUB-MODULES] ________________________________________________________________________________________________________________________
        # backbone
        if 'wavlm' in self.BACKBONE_CFG:
            self.backbone = WavLMModel.from_pretrained(self.BACKBONE_CFG) \
                if self.USE_PRETRAIN else WavLMModel(WavLMConfig.from_pretrained(self.BACKBONE_CFG))

        elif 'wav2vec2' in self.BACKBONE_CFG:
            self.backbone = Wav2Vec2Model.from_pretrained(self.BACKBONE_CFG) \
                if self.USE_PRETRAIN else Wav2Vec2Model(Wav2Vec2Config.from_pretrained(self.BACKBONE_CFG))
        
        elif 'hubert' in self.BACKBONE_CFG:
            self.backbone = HubertModel.from_pretrained(self.BACKBONE_CFG) \
                if self.USE_PRETRAIN else HubertModel(HubertConfig.from_pretrained(self.BACKBONE_CFG))
        else:
            raise NotImplementedError(self.BACKBONE_CFG)
        
        self.N_LAYERS = self.backbone.config.num_hidden_layers if self.N_LAYERS==0 else self.N_LAYERS
        if self.FRZ_PRETRAIN: self.freeze_backbone_modules_()
        
        # speaker representation encoder
        net = ECAPA(
                in_channels=self.backbone.config.hidden_size,
                channels=self.CONV_DIM,
            ); init_weights(net)
        l_pool = LayerAttentionPool(
                in_dim=net.get_out_dim(),
                out_dim=self.HIDDEN_DIM,
                n_layers=self.N_LAYERS+1,
                n_heads=self.N_HEADS,
            )
        t_pool = AttentiveStatisticPool(
                self.HIDDEN_DIM,
                self.POOL_DIM,
            )
        self.encoder = nn.ModuleDict({
            'net': net,
            'l_pool': l_pool,
            't_pool': t_pool
        })
        
        # Loss head
        self.aam_softmax = Multiheaded_AAMsoftmax(
            in_dim=self.POOL_DIM,
            n_heads=self.NUM_LOSS_HEADS,
            n_class=self.NUM_TRAIN_CLASS,
            m=0.2, s=30
        )

        
        # [AUGMENTATION] _____________________________________________________________________________________________________________________
        # Span-out
        self.SPANOUT = config['model']['spanout']
        self.SPANOUT_PROB = config['model']['spanout_prob']
        self.SPAN_MASK_PROB = config['model']['span_mask_prob']
        self.SPAN_MASK_LENGTH = config['model']['span_mask_length']
        
        span_mask_kernel = torch.ones(1, 1, self.SPAN_MASK_LENGTH)
        self.register_buffer('span_mask_kernel', span_mask_kernel)
        
    def freeze_backbone_modules_(self):
        self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad_(False)            

    def forward(self, x:torch.Tensor, length:torch.LongTensor=None, target:torch.LongTensor=None):
        """ 
        Input: 
            x: (B, T)
            length: (B,)
            target: (B,)
        """
        # [Backbone] ______________________________________________________________________________________________        
        if length is not None:
            length = self.backbone._get_feat_extract_output_lengths(length)
            attn_mask = torch.arange(length.max().item(), device=x.device) < length[:, None]
        else:
            length = (torch.ones(x.size(0)) * self.backbone._get_feat_extract_output_lengths(x.size(1))).to(int)
            attn_mask = None
        
        output = self.backbone.forward(x, attention_mask=attn_mask, output_hidden_states=True)
        x = rearrange(torch.stack(output.hidden_states), 'L B T D -> B D L T')[:, :, :self.N_LAYERS+1, :]
        
        # [SpanOut] ______________________________________________________________________________________________
        if self.training:
            x, length = self.spanout_(x, length) # (B, D, L, T >> T')
            
        # [Speaker Encoder] ______________________________________________________________________________________

        # ATTN VAD
        attn_mask = torch.arange(x.size(-1), device=x.device) < length[:, None] if self.training else None
        # x, vad_scale = self.encoder['attn_vad'](x, attn_mask) # (B, D, L, T)
        
        # D2 Block
        x = self.encoder['net'](x) # (B, D->D', L, T)

        # Layer-attention pooling
        x = self.encoder['l_pool'](x) # (B, D, T)
                
        # Layer-attention pooling
        x = self.encoder['t_pool'](x, attn_mask) # (B, D, T)
        
        # [Loss Head] ____________________________________________________________________________________________
        if self.training:
            cls_pred, L = self.aam_softmax(x, target)
            
            # return cls_pred, L + commit_loss
            return cls_pred, L
        
        else:
            x = rearrange(x, 'B (h k) -> B h k', h=self.NUM_LOSS_HEADS)
            
            return F.normalize(x.squeeze(0), p=2, dim=-1)

    def forward_check_(self, B:int):
        # Training forward
        print('[training forward check]')
        self.train()
        target = torch.randint(low=0, high=self.NUM_TRAIN_CLASS, size=(B,))
        length = (torch.rand(B) * 10 * 16000).to(int)
        print('- length:', length / 16000)
        print('- target:', target)
        
        x = torch.rand(B, length.max().item())
        x = F.layer_norm(x, x.size())
        print('- input:', x.size())
        
        cls_pred, L = self.forward(x, length, target)
        print('class prediction:', cls_pred.size())
        print('loss:', L, '\n')
        
        # Evaluation forward
        print('[evaluation forward check]')
        self.eval()
        x = torch.rand(1, length.max().item())
        x = F.layer_norm(x, x.size())
                
        x = self.forward(x)
        print('embedding output:', x.size())
        

    def spanout_(self, x:torch.Tensor, length:torch.LongTensor):
        """ 
        Input:
            x: (B, D, L, T)
            length: (B,)
        Output:
            x: (B, D, L, T') , where T' <= T
            length: (B,)
        """
        if self.SPANOUT:
            B, D, L, T = x.size()
            with torch.no_grad():
                # valid sequence mask: (B, T)
                pad_mask = torch.arange(T, device=x.device) < length[:, None]
                
                # partially span-dropped sequence mask: (B, T)
                spn_mask = torch.rand_like(pad_mask.float(), device=pad_mask.device) < self.SPAN_MASK_PROB # (B, T): randomly selected indice mask (the start of the span to drop)
                spn_mask = F.conv1d(spn_mask.flip(1).unsqueeze(1).float(), 
                                    self.span_mask_kernel, 
                                    padding=self.SPAN_MASK_LENGTH-1).squeeze(1).bool().flip(1)[:,:T] # (B, T): span-mask to drop the representations
                spn_mask = pad_mask & ~spn_mask
                
                # instance-wise augmentation appliance mask: (B,)
                keep_seq = torch.rand(B) > self.SPANOUT_PROB
                spn_mask[keep_seq] = pad_mask[keep_seq]
                
                # the final mask, after concatenating the valid spans: (B, T')
                new_mask = torch.arange(T, device=x.device) < spn_mask.sum(dim=1, keepdim=True)
            
                # __________________________________________________________________________________________________________________________________    
                x = x.masked_scatter_(new_mask[:, None, None, :].expand(-1, D, L, -1), x[spn_mask[:, None, None, :].expand(-1, D, L, -1)])
                length = new_mask.sum(dim=-1)
                
                x = x[..., :length.max().item()]
            
        return x, length



# %%

if __name__ == '__main__':

    from tqdm import tqdm
    from easydict import EasyDict
    from ptflops import get_model_complexity_info

    import time
    import yaml
    import ruamel.yaml

    def hypload(yaml_path:str):
        """ Load '.yaml' file
        
        Args:
            yaml_path (str, pathlib.Path): path to .yaml file

        Returns:
            hyps (dict): python dictionary
        """
        with open(yaml_path, 'r') as f:
            hyps = yaml.load(f, Loader=yaml.FullLoader)
                
        return EasyDict(hyps)


    config = EasyDict()
    config['model'] = hypload('./model-config.yaml')
    # config['model']['backbone_cfg'] = 'microsoft/wavlm-large'  # Example backbone configuration
    config['model']['backbone_cfg'] = 'microsoft/wavlm-base'  # Example backbone configuration

    config['data'] = {'nb_class_train': 1211}  # Example number of training classes
    model = SVmodel(config)
    model.eval()

    # with torch.no_grad():
    #     macs, params = get_model_complexity_info(model, (16000*3,), as_strings=True, backend='pytorch', print_per_layer_stat=True, verbose=False)

    #%%
    device = torch.device(1)
    model.to(device)

    #%%
    # warmup = 10
    # total  = 30
    # batch_size = 32
    
    # model.train()
    # training_times = []
    # for i in tqdm(range(total)):
    #     waveforms = torch.rand(size=(batch_size, 16000*3), device=device)
    #     lengths = torch.ones(batch_size, device=device).long() * 16000 * 3  # 3 seconds of audio
    #     waveforms = F.layer_norm(waveforms, waveforms.size())
    #     lengths = lengths.to(torch.int32)
    #     labels = torch.randint(0, config['data']['nb_class_train'], (batch_size,), device=device)

    #     model.zero_grad()
    #     s_time = time.time()

    #     pred, loss = model(waveforms, lengths, labels)
    #     loss.backward()

    #     e_time = time.time() - s_time

    #     if i < warmup:
    #         pass
    #     else:
    #         training_times.append(e_time)
        
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()
    
    # print('Training time per batch ({:d} samples): {:.04f} seconds'.format(batch_size, np.mean(training_times)))

    
    #%%
    warmup = 200
    total  = 500

    model.eval()
    with torch.no_grad():

        durations = [2, 4, 8, 16, 32]
        for d in durations:

            inference_times = []
            for i in tqdm(range(total)):
                x = torch.rand(size=(1, 16000*d), device=device)
                x = F.layer_norm(x, x.size())

                s_time = time.time()
                x = model(x)
                e_time = time.time() - s_time

                if i < warmup: pass
                else:   inference_times.append(e_time)

            print('Inference time per sample ({:d}s): {:.04f} miliseconds'.format(d, np.mean(inference_times)*1000))
            torch.cuda.empty_cache()

# %%
