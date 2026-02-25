'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet_Influence(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Influence, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if return_features:
            return out, features
        return out

    def get_last_layer_grad(self, x, y, criterion):
        """Returns a flat gradient vector for self.linear params."""
        self.eval()
        self.zero_grad(set_to_none=True)
        out, _ = self.forward(x, return_features=True)
        loss = criterion(out, y)
        grads = torch.autograd.grad(loss, tuple(self.linear.parameters()),
                                    create_graph=False)
        return torch.cat([g.reshape(-1) for g in grads])


    def compute_last_layer_hessian(self, loader, device, damping=1e-4):
        """
        Builds the exact empirical Hessian over the linear classifier.
        H = (1/n) sum_i  (D_i - p_i p_i^T) ⊗ h_i h_i^T
        where h_i are features and p_i are softmax probabilities.
        """
        self.eval()
        H = None
        n = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits, feats = self.forward(x, return_features=True)
                probs = torch.softmax(logits, dim=1) # [B, C]

                for i in range(feats.size(0)):
                    h = feats[i] # [512]
                    p = probs[i] # [C]
                    # Covariance of the softmax output
                    P_cov = torch.diag(p) - torch.outer(p, p)  # [C, C]
                    # Kronecker product for Hessian
                    contrib = torch.kron(P_cov, torch.outer(h, h))  # [C*512, C*512]
                    H = contrib if H is None else H + contrib
                    n += 1

        H = H / n
        # Damping for numerical stability
        H += damping * torch.eye(H.size(0), device=device)
        return H
    

    def influence_score(self, x_train, y_train, x_test, y_test,
                        criterion, H_inv, device):
        """
        I(z_train, z_test) = -grad_test^T @ H_inv @ grad_train
        Positive means removing z_train would HURT performance on z_test.
        Negative means removing z_train would HELP  performance on z_test.
        """
        g_train = self.get_last_layer_grad(x_train, y_train, criterion)
        g_test  = self.get_last_layer_grad(x_test,  y_test,  criterion)
        return -(g_test @ H_inv @ g_train).item()
    
    def compute_influence_matrix(self, dataloader, H_inv, device):
        """
        Precomputes the N x N influence matrix C where C[j, i] is the 
        change in the LiRA statistic (scaled logit) for point j if 
        point i is added to the training set.
        """
        self.eval()
        
        # 1. Collect all features and true labels
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                logits, feats = self.forward(x, return_features=True)
                all_feats.append(feats)
                all_labels.append(y.to(device))
                
        all_feats = torch.cat(all_feats, dim=0)    # [N, 512]
        all_labels = torch.cat(all_labels, dim=0)  # [N]
        N = all_feats.size(0)

        with torch.no_grad():
            logits = self.linear(all_feats)        # [N, C]
            probs = torch.softmax(logits, dim=1)   # [N, C]
            
            y_onehot = F.one_hot(all_labels, num_classes=self.num_classes).float()
            

            d_logits = probs - y_onehot            # [N, C]
            
            grad_W = torch.einsum('ni,nj->nij', d_logits, all_feats) 
            
            grad_b = d_logits
            
            G = torch.cat([grad_W.reshape(N, -1), grad_b], dim=1)
        
        C = -(G @ H_inv @ G.T) / N  # The 1/N comes from the influence function definition
        
        return C
    
    
    def get_lira_statistics(self, dataloader, device):
        """
        Computes the base LiRA statistic for every point in the dataloader.
        Returns a 1D tensor of length N containing the scaled logits.
        """
        self.eval()
        t_bases = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                # Forward pass to get raw logits [Batch_Size, Num_Classes]
                logits = self.forward(x, return_features=False)
                
                # Get the probabilities using softmax
                probs = torch.softmax(logits, dim=1)
                
                # Extract the probability of the *true* class for each image
                # gather() pulls the probability at the index specified by y
                p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)
                
                # Clamp p_true to avoid log(0) or log(1) resulting in infinity
                p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
                
                # Compute the LiRA scaled logit: log( p / (1-p) )
                t = torch.log(p_true / (1.0 - p_true))
                
                t_bases.append(t)
                
        # Concatenate into a single 1D tensor of size N
        return torch.cat(t_bases, dim=0)




def ResNet18_Influence():
    return ResNet_Influence(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18_Influence()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()