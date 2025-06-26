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
    

class RecUnet(nn.Module):
    def __init__(self, unet, filters=[16,32,64,128,256,32]):
        super(RecUnet, self).__init__()
        self.unet = unet
        self.projector_1 = Projector(in_dim=filters[0], out_dim=filters[0], pool_param=(1,1))
        self.projector_2 = Projector(in_dim=filters[1], out_dim=filters[1], pool_param=(1,1))
        self.projector_3 = Projector(in_dim=filters[2], out_dim=filters[2], pool_param=(1,1))
        self.projector_4 = Projector(in_dim=filters[3], out_dim=filters[3], pool_param=(1,1))

    
    def forward(self, x):
        rec = None
        latent = None
        rec = self.unet(x)
        x = self.unet.conv_0(x)
        out1 = self.unet.down_1(x)
        out2 = self.unet.down_2(out1)
        out3 = self.unet.down_3(out2)
        out4 = self.unet.down_4(out3)

        # get the multi-scale embeddings
        out1 = self.projector_1(out1)
        out2 = self.projector_2(out2)
        out3 = self.projector_3(out3)
        out4 = self.projector_4(out4)
        latent = torch.cat([out1, out2, out3, out4], dim=-1) 

        return rec, latent