import torch
import torch.nn as nn
from modules.tailored_decoder import EncoderDecoder
import torchvision.models as models
from modules.pvtv2 import pvt_v2_b2
import math
import torch.nn.functional as F
from info_nce import InfoNCE
import random
import torchvision
from einops import repeat, rearrange


class Model(nn.Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        self.L = (8-self.args.kernel)**2
        # text encoder
        self.word_embd = nn.Embedding(self.encoder_decoder.vocab_size + 1, args.d_model)
        self.word_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.d_model, nhead=8), num_layers=3)
        self.word_mlp = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Tanh(), nn.Linear(args.d_model, self.L))
        self.att_embed_report = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, args.d_model), nn.Dropout(args.drop_prob_lm))

        pe = torch.zeros(120, args.d_model)
        position = torch.arange(0, 120).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, args.d_model, 2).float() * -(math.log(10000.0) / args.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # image encoder
        if args.visual_extractor == "densenet121":
            model = getattr(models, args.visual_extractor)(pretrained=args.pretrained)
            self.vision = model.features
            self.att_feat_size = 1024

        elif args.visual_extractor == "resnet18":
            model = getattr(models, args.visual_extractor)(pretrained=args.pretrained)
            modules = nn.ModuleList(model.children())[:-2]
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 512

        elif args.visual_extractor == "resnet50":
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 512, bias=False) # projection head
            state_dict = torch.load("/public/home/jw12138/.cache/torch/hub/checkpoints/pytorch_model.bin")
            diction = {}
            for key in state_dict:
                if key.split(".")[0]== "vision_model":
                    diction_key = key.replace("vision_model.model.","")
                    diction[diction_key] = state_dict[key]
            model.load_state_dict(diction, strict=False)
            modules = nn.ModuleList(model.children())[:-2]
            self.vision = nn.Sequential(*modules) 
            self.att_feat_size = 2048
        else:
            self.vision = pvt_v2_b2()  # [64, 128, 320, 512]
            path = '/public/home/jw12138/.cache/torch/hub/checkpoints/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.vision.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.vision.load_state_dict(model_dict)
            self.att_feat_size = 512

        d_middle = 1024
        self.cnn = nn.Conv2d(self.att_feat_size, d_middle, self.args.kernel, stride=1)
        self.att_embed_image = nn.Sequential(nn.Linear(d_middle, args.d_model), nn.ReLU(),nn.Linear(args.d_model, args.d_model), nn.Dropout(args.drop_prob_lm))

    def forward(self, images, targets=None, tok=None, mode='train', tags=0, epoch_id=0):
        # in training, sample_v and sample_t

        if mode == 'train':
            if self.args.visual_extractor == "pvt":
                patch_feats = self.cnn(self.vision(images)[3])
            else:
                patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  
            att_feats_0 = self.att_embed_image(patch_feats_f)
            
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))

            if epoch_id % self.args.RoundGap == 0:
                a = int(epoch_id/self.args.RoundGap)
                feats = att_feats_0
                replace = random.sample(range(0, self.L), a)
                for i in replace:
                    feats[:, i, :] = sturctured_emb_0[:, i, :]
                output_rev = self.encoder_decoder(feats, targets, mode='forward')
                return output_rev
            
            if epoch_id % self.args.RoundGap == self.args.RoundGap -1:
                a = int((epoch_id+1)/self.args.RoundGap)
                feats = sturctured_emb_0
                replace = random.sample(range(0, self.L), a)
                for i in replace:
                    feats[:, i, :] = att_feats_0[:, i, :]
                output_ret = self.encoder_decoder(feats, targets, mode='forward')
                return output_ret
            
            else:
                output_t = self.encoder_decoder(sturctured_emb_0, targets, mode='forward')
                output_v = self.encoder_decoder(att_feats_0, targets, mode='forward')
                # loss_mse = self.mse_loss(att_feats_0, sturctured_emb_0) 
                # loss_kl = self.kl_loss(F.log_softmax(att_feats_0, dim=-1), F.softmax(sturctured_emb_0, dim=-1) ) 
                return output_t, output_v

            
        elif mode == 'sample_v':
            if self.args.visual_extractor == "pvt":
                patch_feats = self.cnn(self.vision(images)[3])
            else:
                patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  
            att_feats_0 = self.att_embed_image(patch_feats_f)
            output_v, probabilities = self.encoder_decoder(att_feats_0, att_feats_0, mode='sample')
            return output_v

        elif mode == 'sample_t':
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))
            output_t, probabilities = self.encoder_decoder(sturctured_emb_0, sturctured_emb_0, mode='sample')
            return output_t


