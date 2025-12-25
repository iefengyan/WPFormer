class FeatureInjectionModule(nn.Module):
    def __init__(self,num_channels):
        super(FeatureInjectionModule,self).__init__()
        channels = num_channels
        self.fft = Freprocess(channels)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
    def forward(self,msf,panf):
        fre = self.fft(msf,panf)
        # msf = self.conv(msf)
        spa = self.fuse(torch.cat([fre,msf],1))
        fuse = spa+msf
        return fuse
class StripAtten(nn.Module):
    def __init__(self, in_channels):
        super(StripAtten, self).__init__()

        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))

        self.k1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.q1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.k2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.q2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.pro = nn.Conv2d(in_channels, in_channels, kernel_size=1)


    def forward(self, kv, q):

        bs, c, H, W = kv.size()
        # kv1=self.pool1(kv)
        q1 = self.q1(q).flatten(2)
        k1 = self.k1(kv).flatten(2)
        v1 = self.v1(kv).flatten(2)

        s_atten = torch.matmul(q1, k1.transpose(-2, -1))
        s_atten_score = F.softmax(s_atten, dim=-1)
        out = torch.matmul(s_atten_score, v1).view(bs,c,H,W)

        return out

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class DDFusion(nn.Module):
    def __init__(self, in_channels,dct_h=8):
        super(DDFusion, self).__init__()
        self.frenum=8

        # self.dct = LFE(in_channels, dct_h=dct_h, dct_w=dct_h, frenum=8)
        # self.idct = LFE(in_channels, dct_h=dct_h, dct_w=dct_h, frenum=8)


        self.fused=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1,bias=False),
            Norm(in_channels),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.ca1 = StripAtten(64)
        self.ca2 = StripAtten(64)

        self.sa = StripAtten(64)
        self.energy= simam_module()

    def forward(self, x, y):
        bs, c, H, W = y.size()
        # 1st layer
        # x_dct = self.dct(x)
        # y_dct = self.dct(y)
        #
        #
        # xy_dct = self.fused(x_dct+y_dct)
        #
        # attn1 = self.ca1(kv=xy_dct,q=x_dct)
        # attn2 = self.ca2(kv=xy_dct,q=y_dct)
        #
        #
        # attn = attn2 + attn1
        # attn = self.sa(kv=attn,q=attn)
        #
        #
        # attn = torch.sum(attn, dim=[2,3], keepdim=True)
        #
        # x = self.fc1(attn) * x + x
        # y = self.fc2(attn) * y + y


        x = F.upsample(x, size=(H, W), mode='bilinear')
        out=x+y
        return out
