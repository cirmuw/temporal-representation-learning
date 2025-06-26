import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, pool_param):
        super(Projector, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_param)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.projector(x)
    

class MTANRecUnet(nn.Module):
    def __init__(self, unet, filters=[16,32,64,128,256,32]):
        super(MTANRecUnet, self).__init__()
        self.unet = unet
        self.projector_1 = Projector(in_dim=filters[0], out_dim=filters[0], pool_param=(1,1))
        self.projector_2 = Projector(in_dim=filters[1], out_dim=filters[1], pool_param=(1,1))
        self.projector_3 = Projector(in_dim=filters[2], out_dim=filters[2], pool_param=(1,1))
        self.projector_4 = Projector(in_dim=filters[3], out_dim=filters[3], pool_param=(1,1))
        self.attention_blocks = nn.ModuleList([self.att_layer([filters[i], filters[i], filters[i]]) for i in range(4)])

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], channel[2], kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        return att_block
    
    def mask_feature_maps(self, encoder_features, i):
        
        attn_mask = self.attention_blocks[i](encoder_features)
        refined_feature = attn_mask * encoder_features  # Apply attention
        return refined_feature, None
    
    def forward(self, x):
        rec = None
        latent = None
        rec = self.unet(x)
        x = self.unet.conv_0(x)
        out1 = self.unet.down_1(x)
        out2 = self.unet.down_2(out1)
        out3 = self.unet.down_3(out2)
        out4 = self.unet.down_4(out3)
        
        # apply attention masks
        out1 = self.mask_feature_maps(out1, 0)
        out2 = self.mask_feature_maps(out2, 1)
        out3 = self.mask_feature_maps(out3, 2)
        out4 = self.mask_feature_maps(out4, 3)
        
        # get the multi-scale embeddings
        out1 = self.projector_1(out1)
        out2 = self.projector_2(out2)
        out3 = self.projector_3(out3)
        out4 = self.projector_4(out4)
        latent = torch.cat([out1, out2, out3, out4], dim=-1) 

        return rec, latent