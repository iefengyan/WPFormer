import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.pvtv2 import pvt_v2_b2,pvt_v2_b4

from typing import Optional
from torch import nn, Tensor
import math, copy
from model.position_encoding import PositionEmbeddingSine
from model.transformer import Transformer, SelfAttentionLayer, FFNLayer, MLP, _get_activation_fn, _get_clones

from model import wavelet

class DDFusion(nn.Module):
    def __init__(self, in_channels, dct_h=8):
        super(DDFusion, self).__init__()

    def forward(self, x, y):
        bs, c, H, W = y.size()

        x = F.upsample(x, size=(H, W), mode='bilinear')
        out = x + y
        return out


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def _reset_parameters(self):
        for p in self.parameters():
            print(p.dim())
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask)


        return tgt2



    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=memory,
                                   value=memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MSCW(nn.Module):
    def __init__(self, d_model=64):
        super(MSCW, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
        )
        self.local_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.global_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.linear = nn.Linear(d_model,d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.local_attn(x) + self.global_attn(pool)
        attn = self.sigmoid(attn)
        return attn


class SegHead(nn.Module):
    def __init__(self, channel):
        super(SegHead, self).__init__()
        self.conv = convbnrelu(channel, channel, k=1, s=1, p=0)

        self.decoder_norm = nn.LayerNorm(channel)
        self.class_embed = nn.Linear(channel, 1)
        self.mask_embed = MLP(channel, channel, channel, 3)
        # self.num_heads = 8

    def forward(self, output, mask_features,attn_mask_target_size):

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        mask_features = self.conv(mask_features)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = None

        return outputs_class, outputs_mask, attn_mask




class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )

    def forward(self, x):
        return self.conv(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.norm1 = nn.LayerNorm(d_model)


        self.pool = wavelet.WavePool(d_model)
        self.self_attn1 = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.mscw1 = MSCW(d_model=d_model)

        self.proto_size = 16
        self.conv3x3 = DSConv3x3(d_model, d_model)
        self.Mheads = nn.Linear(d_model, self.proto_size, bias=False)
        self.mscw2 = MSCW(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)



    def forward(self, query, key, value, attn_mask=None):

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        b, n1, c = value.size()
        hw = int(math.sqrt(n1))

        feat = key.transpose(1, 2).view(b, c, hw, hw)
        #
        LL, HL, LH, HH = self.pool(feat)
        high_fre = HL + LH + HH
        low_fre = LL
        high_fre = high_fre.flatten(2).transpose(1, 2)
        low_fre = low_fre.flatten(2).transpose(1, 2)
        wei = self.mscw1(high_fre+low_fre)

        fre = wei*high_fre+low_fre
        query1 =query
        x1 = self.self_attn1(query=query1, key=fre, value=fre, attn_mask=None)[0]
        x1 = self.norm1(x1+query1)

        # channel attention
        feat = self.conv3x3(feat).flatten(2).transpose(1, 2)
        multi_heads_weights = self.Mheads(feat)
        multi_heads_weights = multi_heads_weights.view((b, n1, self.proto_size))
        multi_heads_weights = F.softmax(multi_heads_weights, dim=1)
        protos = multi_heads_weights.transpose(-1, -2) @ key
        query2 = query

        attn = self.mscw2(protos+query2)
        x2 = query2 * attn + query2
        x2 = self.norm2(x2)

        x = x1+x2


        return x.transpose(0, 1)


class WPFormer(nn.Module):
    def __init__(self, channel=64, num_queries=16):
        super(WPFormer, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'D:\yanfeng\Paper Code\CVPR2025\WPFormer\model\pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)

        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.latlayer1 = BasicConv2d(channel, channel, 1)
        self.latlayer2 = BasicConv2d(channel, channel, 1)
        self.latlayer3 = BasicConv2d(channel, channel, 1)
        self.latlayer4 = BasicConv2d(channel, channel, 1)

        self.outconv1 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv2 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv3 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv4 = convbnrelu(channel, channel, k=3, s=1, p=1)

        self.fusion1 = DDFusion(channel)
        self.fusion2 = DDFusion(channel)
        self.fusion3 = DDFusion(channel)
        self.fusion4 = DDFusion(channel)

        self.query_embed = nn.Embedding(num_queries, channel)

        # positional encoding
        N_steps = channel // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.num_heads = 8
        self.transformer_self_attention_layers = nn.ModuleList()
        self.task_cross_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_feature_levels = 3

        for _ in range(self.num_feature_levels):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=channel,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
               CrossAttentionLayer(
                    d_model=channel,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )


        self.level_embed = nn.Embedding(self.num_feature_levels, channel)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.input_proj.append(convbnrelu(channel, channel, k=1, s=1, p=0))
        self.mask_features = convbnrelu(channel, channel, k=3, s=1, p=1)

        self.class_transformer = Transformer(
            d_model=channel,
            dropout=0.1,
            nhead=self.num_heads,
            dim_feedforward=2048,
            num_encoder_layers=0,
            num_decoder_layers=2,
            normalize_before=False,
            return_intermediate_dec=False,
        )
        self.class_input_proj = convbnrelu(channel, channel, k=1, s=1, p=0)
        self.SegHeads = nn.ModuleList()
        for _ in range(self.num_feature_levels+1):
            self.SegHeads.append(SegHead(channel))


    def upsample_add(self, x, y):
        bs, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y
    #
    def forward_prediction_heads(self, output, mask_features):
        bs = output.size()[1]
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = None

        return outputs_class, outputs_mask, attn_mask

    def forward(self, x):
        image_shape = x.size()[2:]
        bs = x.size()[0]
        pvt = self.backbone(x)

        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_t = self.Translayer1_1(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        # FPN
        d3 = self.outconv3(self.fusion1(x4_t, self.latlayer3(x3_t)))
        d2 = self.outconv2(self.fusion2(d3, self.latlayer2(x2_t)))
        d1 = self.outconv1(self.fusion3(d2, self.latlayer1(x1_t)))

        x = [x4_t, d3, d2]

        src = []
        pos = []
        size_list = []
        for i in range(self.num_feature_levels):
            # print(x[i].size())
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        mask_features = self.mask_features(d1)

        out_t, _ = self.class_transformer(self.pe_layer(mask_features), None, self.query_embed.weight,
                                          self.class_input_proj(mask_features), None)

        out_t = out_t[0]
        output = out_t
        predictions_mask = []

        outputs_class, outputs_mask, attn_mask = self.SegHeads[0](output, mask_features, attn_mask_target_size=size_list[0])


        predictions_mask.append(self.semantic_inference(outputs_class, outputs_mask))

        for i in range(self.num_feature_levels):
            level_index = i % self.num_feature_levels

            # attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # fres_feat.append(fre)

            outputs_class, outputs_mask, attn_mask = self.SegHeads[i+1](output, mask_features,attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_mask.append(self.semantic_inference(outputs_class, outputs_mask))



        mask = predictions_mask[1]+predictions_mask[2]+ predictions_mask[3]

        predictions_mask.append(mask)

        for i in range(self.num_feature_levels + 2):
            predictions_mask[i] = F.interpolate(predictions_mask[i], size=image_shape, mode='bilinear')

        return predictions_mask

    def semantic_inference(self, mask_cls, mask_pred):

        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg




import time
import numpy as np
if __name__ == '__main__':
    #images = torch.rand(2, 3, 224, 224)
    # images = torch.rand(1, 3, 320, 320).cuda(0)
    model = WPQENet()
    # model.load_state_dict(torch.load("D:\yanfeng\project\\model.pth"))
    from mmcv.cnn import get_model_complexity_info

    if torch.cuda.is_available():
        net = model.cuda()
    net.eval()
    flops, params = get_model_complexity_info(net, input_shape=(3, 384, 384))
    print(flops)
    print(params)

    bc = 1
    dump_x = torch.randn(bc, 3, 384, 384).cuda()

    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        with torch.no_grad():
            start_time = time.time()
            res = net(dump_x)
            end_time = time.time()

        running_frame_rate = (1 * float((bc / (end_time - start_time))))
        # print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))



