import torch
import torch.nn as nn
import torch.nn.functional as F

from hawtorch.nn import fcbr, conv1dbr, conv2dbr


"""
Capsule Net
"""

def squash(x, dim=-1, epsilon=1e-8):
    norm = (x**2).sum(dim=dim, keepdim=True)
    x = norm / (norm + 1) * x / (torch.sqrt(norm) + epsilon)
    return x


class PrimaryCapsules(nn.Module):
    def __init__(self, in_features, capsules_num, capsules_dim):
        super(PrimaryCapsules, self).__init__()
        self.in_features = in_features
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim

        self.conv = nn.Conv2d(in_features, capsules_num*capsules_dim, kernel_size=9, stride=2)

    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x) # [B, 32*8, 6, 6]
        """
        Since all capsules use the same convolution operations, just do once and reshape.
        """
        #x = x.view(batch_size, self.capsules_num, self.capsules_dim, -1)
        #x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, self.capsules_dim)
        x = squash(x)
        return x


class RoutingCapsules(nn.Module):
    """
    input
        capsules_num: new feature, num to duplicate
        capsules_dim: new feature, each capsules' dim
        in_capsules_num: last layer's capsules_num
        in_capsules_dim: last layer's capsules_dim
    """
    def __init__(self, capsules_num, capsules_dim, in_capsules_num, in_capsules_dim, num_iterations=3):
        super(RoutingCapsules, self).__init__()
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim
        self.in_capsules_num = in_capsules_num
        self.in_capsules_dim = in_capsules_dim
        self.num_iterations = num_iterations

        self.W = nn.Parameter(torch.randn(1, capsules_num, in_capsules_num, in_capsules_dim, capsules_dim))
        


    def forward(self, x):
        batch_size = x.shape[0] # [B, in_capsules_num, in_capsules_dim]
        x = x.unsqueeze(1).unsqueeze(3) # [B, 1, 32*6*6, 1, 8]
        #print(self.W.shape, x.shape)
        u_hat = x @ self.W # [B, 10, 32*6*6, 1, 16]
        
        b = torch.zeros_like(u_hat).to(x.device)

        for i in range(self.num_iterations - 1):
            """
            Softmax is applied on all of the input capsules, to calculate probs.
            """
            c = F.softmax(b, dim=2) # [B, 10, 32*6*6, 1, 16]
            s = (c * u_hat).sum(dim=2, keepdim=True) # [B, 10, 1, 1, 16]
            v = squash(s)
            uv = u_hat * v # [B, 10, 32*6*6, 16, 1]
            b = b + uv 
            # never use `b += uv` !!! <1h+ wasted>

        c = F.softmax(b, dim=2)
        s = (c * u_hat).sum(dim=2, keepdim=True)
        v = squash(s)
        v = v.squeeze()

        return v # [B, capsules_num, capsules_dim]

class Decoder(nn.Module):
    def __init__(self, num_points, num_capsules, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.seq = nn.Sequential(
            nn.Linear(num_capsules * in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
            #nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.seq(x)
        return x
        


class CapsNet(nn.Module):
    def __init__(self, args):
        super(CapsNet, self).__init__()
        self.num_points = args["num_points"]
        self.in_channels = args["in_channels"]
        self.num_classes = args["num_classes"]
        self.out_channels = args["out_channels"]

        self.conv = nn.Conv2d(1, 256, 9, 1)
        self.relu = nn.ReLU(inplace=True)

        self.primary = PrimaryCapsules(in_features=256, 
                                       capsules_num=32, 
                                       capsules_dim=8)

        self.route = RoutingCapsules(capsules_num=self.num_classes, 
                                      capsules_dim=self.out_channels, 
                                      in_capsules_num=32*6*6, 
                                      in_capsules_dim=8, 
                                      num_iterations=3)

        self.decoder = Decoder(num_points=self.num_points, 
                               num_capsules=self.num_classes, 
                               in_channels=self.out_channels, 
                               out_channels=self.in_channels)

    def forward(self, x, y=None):
        x = F.relu(self.conv(x), inplace=True)
        x = self.primary(x)
        x = self.route(x) # [B, num_classes, out_channels]
        
        logits = (x**2).sum(dim=-1).sqrt()
        logits = F.softmax(logits, dim=-1) # [B, num_classes]

        # testing mode
        if y is None:
            preds = logits.max(dim=1)[1] # argmax, [B,]
            y = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=preds) # [B, num_classes, 1]
        
        recons = self.decoder(x * y.unsqueeze(2))
        return logits, recons


