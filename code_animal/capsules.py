import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model import *

class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out


class ConvCaps(nn.Module):    # raw
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, B, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        # oh = ow = int((h - K + 1) / stride)
        oh = ow = int((h - K) / stride + 1)
        idxs = [[(h_idx + k_idx) \
                for k_idx in range(0, K)] \
                for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
            # return out
            return a_out, p_out
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            pout, out = self.caps_em_routing(v, a_in, self.C, self.eps)

            return out, pout



class CapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=True, w_shared=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.class_caps(x)
        return x


def capsules(**kwargs):
    """Constructs a CapsNet model.
    """
    model = CapsNet(**kwargs)
    return model



class MyCapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet, self).__init__()
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x):
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet1(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A0=512), the feature maps change as follows:
    0. squeeze feature map channel: A=256
        (_, 512, 7, 7) -> 1x1 filter, 256 out channels, stride 1, no padding
        x -> (_, 256, 7, 7)
    1. PrimaryCaps: B=32
        (_, 256, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet1, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(512, A, kernel_size=1, stride=1),
                nn.BatchNorm2d(A),
                nn.ReLU(inplace=True)
                )
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet2(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A0=512), the feature maps change as follows:
    0. downsample feature map: A=512
        (_, 512, 7, 7) -> maxpool, 2x2 filter, stride 2
        x -> (_, 512, 3, 3)
    1. PrimaryCaps: B=32
        (_, 512, 3, 3) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    2. ConvCaps1: C=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    3. ConvCaps2: D=32
        (_, 1, 1, 32x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, 1, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x):
        x = self.pool1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, pose = self.class_caps(x)
        return x, pose




class MyCapsNet_cam(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(2*C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        a = torch.cat([a1, a2], dim=3)
        p = torch.cat([p1, p2], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps2(x)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam1(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x1024(A=1024), the feature maps change as follows:
    1. PrimaryCaps: B=64
        (_, 1024, 7, 7) -> 1x1 filter, 64 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 64x4x4), activation: (_, 7, 7, 64)
    2. ConvCaps1: C=64
        (_, 7, 7, 64x(4x4+1)) -> 3x3 filters, 64 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 64x4x4), activation: (_, 3, 3, 64)
    3. ConvCaps2: D=32
        (_, 3, 3, 64x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam1, self).__init__()
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2),1)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam2(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam2, self).__init__()
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x = x1 + x2
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam3(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam3, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps3 = PrimaryCaps(2*A, 2*B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(2*B, D, K, P, stride=2, iters=iters)
        self.conv_caps2_2 = ConvCaps(2*C, D, 1, P, stride=1, iters=iters)
        self.conv_caps3 = ConvCaps(2*D, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        xc = torch.cat((x1, x2),1)
        xc = self.primary_caps3(xc)
        ac, pc = self.conv_caps2_1(xc)
        
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        a = torch.cat([a1, a2], dim=3)
        p = torch.cat([p1, p2], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps2_2(x)
        
        aw = torch.cat([ac, xa], dim=3)
        pw = torch.cat([pc, xp], dim=3)
        xw = torch.cat([pw, aw], dim=3)
        xwa, xwp = self.conv_caps3(xw)
        xw = torch.cat([xwp, xwa], dim=3)
        xw, posew = self.class_caps(xw)
        return xw, posew


class MyCapsNet_cam4(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        G: number of groups, G = 2
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam4, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D//2, K, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D//2, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        
        N,H,W,CPP = p1.shape
        p11 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,:16,:], p2.view(N,3,3,32,16)[:,:,:,:16,:]),3).view(N,3,3,32*16)
        a11 = torch.cat((a1[:,:,:,:16], a2[:,:,:,:16]),3)
        x11 = torch.cat([p11, a11], dim=3)
        xa1, xp1 = self.conv_caps2_1(x11)
        
        p22 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,16:,:], p2.view(N,3,3,32,16)[:,:,:,16:,:]),3).view(N,3,3,32*16)
        a22 = torch.cat((a1[:,:,:,16:], a2[:,:,:,16:]),3)
        x22 = torch.cat([p22, a22], dim=3)
        xa2, xp2 = self.conv_caps2_2(x22)
        # xa2, xp2 = self.conv_caps2_1(x22)
        
        a = torch.cat([xa1, xa2], dim=3)
        p = torch.cat([xp1, xp2], dim=3)
        x = torch.cat([p, a], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam5(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        G: number of groups, G = 2
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam5, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D//4, K, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D//4, K, P, stride=1, iters=iters)
        self.conv_caps2_3 = ConvCaps(2*C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(48, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        
        N,H,W,CPP = p1.shape
        a11 = torch.cat((a1[:,:,:,:16], a2[:,:,:,:16]),3)
        p11 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,:16,:], p2.view(N,3,3,32,16)[:,:,:,:16,:]),3).view(N,3,3,32*16)
        x11 = torch.cat([p11, a11], dim=3)
        xa1, xp1 = self.conv_caps2_1(x11)
        
        a22 = torch.cat((a1[:,:,:,16:], a2[:,:,:,16:]),3)
        p22 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,16:,:], p2.view(N,3,3,32,16)[:,:,:,16:,:]),3).view(N,3,3,32*16)
        x22 = torch.cat([p22, a22], dim=3)
        xa2, xp2 = self.conv_caps2_2(x22)
        
        a33 = torch.cat((a1, a2),dim=3)
        p33 = torch.cat((p1, p2), dim=3)
        x33 = torch.cat([p33, a33], dim=3)
        xa3, xp3 = self.conv_caps2_3(x33)
        
        a = torch.cat([xa1, xa2, xa3], dim=3)
        p = torch.cat([xp1, xp2, xp3], dim=3)
        x = torch.cat([p, a], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam6(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        G: number of groups, G = 4
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam6, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C//2, D//8, K, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C//2, D//8, K, P, stride=1, iters=iters)
        self.conv_caps2_4 = ConvCaps(C//2, D//8, K, P, stride=1, iters=iters)
        self.conv_caps2_5 = ConvCaps(C//2, D//8, K, P, stride=1, iters=iters)
        self.conv_caps2_3 = ConvCaps(2*C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(48, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        
        IC = 32 // 4
        N,H,W,CPP = p1.shape
        a11 = torch.cat((a1[:,:,:,:IC], a2[:,:,:,:IC]),3)
        p11 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,:IC,:], p2.view(N,3,3,32,16)[:,:,:,:IC,:]),3).view(N,3,3,2*IC*16)
        x11 = torch.cat([p11, a11], dim=3)
        xa1, xp1 = self.conv_caps2_1(x11)
        
        a22 = torch.cat((a1[:,:,:,IC:2*IC], a2[:,:,:,IC:2*IC]),3)
        p22 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,IC:2*IC,:], p2.view(N,3,3,32,16)[:,:,:,IC:2*IC,:]),3).view(N,3,3,2*IC*16)
        x22 = torch.cat([p22, a22], dim=3)
        xa2, xp2 = self.conv_caps2_2(x22)
        
        a44 = torch.cat((a1[:,:,:,2*IC:3*IC], a2[:,:,:,2*IC:3*IC]),3)
        p44 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,2*IC:3*IC,:], p2.view(N,3,3,32,16)[:,:,:,2*IC:3*IC,:]),3).view(N,3,3,2*IC*16)
        x44 = torch.cat([p44, a44], dim=3)
        xa4, xp4 = self.conv_caps2_4(x44)
        
        a55 = torch.cat((a1[:,:,:,3*IC:], a2[:,:,:,3*IC:]),3)
        p55 = torch.cat((p1.view(N,3,3,32,16)[:,:,:,3*IC:,:], p2.view(N,3,3,32,16)[:,:,:,3*IC:,:]),3).view(N,3,3,2*IC*16)
        x55 = torch.cat([p55, a55], dim=3)
        xa5, xp5 = self.conv_caps2_5(x55)
        
        a33 = torch.cat((a1, a2),dim=3)
        p33 = torch.cat((p1, p2), dim=3)
        x33 = torch.cat([p33, a33], dim=3)
        xa3, xp3 = self.conv_caps2_3(x33)
        
        a = torch.cat([xa1, xa2, xa4, xa5, xa3], dim=3)
        p = torch.cat([xp1, xp2, xp4, xp5, xp3], dim=3)
        x = torch.cat([p, a], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam7(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam7, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters) #branch1,3*3
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=1, iters=iters) #branch1,5*5
        self.conv_caps2_1 = ConvCaps(B, C, K, P, stride=2, iters=iters) #branch2,3*3
        self.conv_caps2_2 = ConvCaps(B, C, K, P, stride=1, iters=iters) #branch2,5*5
        self.conv_caps3_1 = ConvCaps(2*C, D, 1, P, stride=1, iters=iters) #for 3*3
        self.conv_caps3_2 = ConvCaps(2*C, D, 1, P, stride=2, iters=iters) #for 5*5
        self.conv_caps4 = ConvCaps(2*D, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1_1, p1_1 = self.conv_caps1_1(x1)
        a1_2, p1_2 = self.conv_caps1_2(x1)
        x2 = self.primary_caps2(x2)
        a2_1, p2_1 = self.conv_caps2_1(x2)
        a2_2, p2_2 = self.conv_caps2_2(x2)
        
        a1 = torch.cat([a1_1, a2_1], dim=3)
        p1 = torch.cat([p1_1, p2_1], dim=3)
        x3 = torch.cat([p1, a1], dim=3)
        xa3, xp3 = self.conv_caps3_1(x3)
        a2 = torch.cat([a1_2, a2_2], dim=3)
        p2 = torch.cat([p1_2, p2_2], dim=3)
        x5 = torch.cat([p2, a2], dim=3)
        xa5, xp5 = self.conv_caps3_2(x5)
        
        a = torch.cat([xa3, xa5], dim=3)
        p = torch.cat([xp3, xp5], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps4(x)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam9(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam9, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 576),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=4)
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps_attention(2*C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        aw = torch.cat((x1,x2),1)
        aw = self.avgpool(aw)
        aw = torch.flatten(aw, 1)
        aw = self.fc1(aw).unsqueeze(2)
        
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        a = torch.cat([a1, a2], dim=3)
        p = torch.cat([p1, p2], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps2(x, aw)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam10(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1_1/2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2_1/2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ConvCaps3_1: F=32
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    5. ConvCaps3_2: F=32
        (_, 3, 3, 32x2x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    6. ClassCaps: E=2
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        F: output channels of 3nd conv caps, F=D
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam10, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.conv_caps3_1 = ConvCaps(D*2, D, 1, P, stride=1, iters=iters)
        self.conv_caps3_2 = ConvCaps(D*2, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D*2, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1_1, p1_1 = self.conv_caps1_1(x1)     # 3*3
        x1_2 = torch.cat([p1_1, a1_1], dim=3)
        a1_2, p1_2 = self.conv_caps2_1(x1_2)    # 1*1
        
        x2 = self.primary_caps2(x2)
        a2_1, p2_1 = self.conv_caps1_2(x2)    # 3*3
        x2_2 = torch.cat([p2_1, a2_1], dim=3)
        a2_2, p2_2 = self.conv_caps2_2(x2_2)    # 1*1
        
        a1 = torch.cat([a1_2, a2_2], dim=3)    # 1*1
        p1 = torch.cat([p1_2, p2_2], dim=3)
        x11 = torch.cat([p1, a1], dim=3)
        xa1, xp1 = self.conv_caps3_1(x11)
        a2 = torch.cat([a1_1, a2_1], dim=3)    # 3*3
        p2 = torch.cat([p1_1, p2_1], dim=3)
        x22 = torch.cat([p2, a2], dim=3)
        xa2, xp2 = self.conv_caps3_2(x22)
        
        xa = torch.cat([xa1, xa2], dim=3)
        xp = torch.cat([xp1, xp2], dim=3)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam11(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    3. ConvCaps2: D=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam11, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(2*C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)
        self.softmax = nn.Softmax(dim=1)
        self.fusionfc = nn.Linear(6, 2)

    def forward(self, x1, x2, o1, o2):
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        a = torch.cat([a1, a2], dim=3)
        p = torch.cat([p1, p2], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps2(x)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        
        o1 = self.softmax(o1)
        o2 = self.softmax(o2)
        o = torch.cat((o1, o2, x), 1)
        o = self.fusionfc(o)
        
        return x, pose, o


class MyCapsNet_cam12(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2.1 ConvCaps1_1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
        ConvCaps2_1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    2.2 ConvCaps1_2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
        ConvCaps2_2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    3. ConvCaps3_1: D=32
        (_, 3, 3, 32x2x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
       ConvCaps3_2: D=32
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam12, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D, 7, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D, 7, P, stride=1, iters=iters)
        self.conv_caps3_1 = ConvCaps(D*2, D, K, P, stride=1, iters=iters)
        self.conv_caps3_2 = ConvCaps(D*2, D, 1, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D*2, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1_1, p1_1 = self.conv_caps1_1(x1)    # 3*3
        a2_1, p2_1 = self.conv_caps2_1(x1)    # 1*1
        x2 = self.primary_caps2(x2)
        a1_2, p1_2 = self.conv_caps1_2(x2)
        a2_2, p2_2 = self.conv_caps2_2(x2)
        a1 = torch.cat([a1_1, a1_2], dim=3)
        p1 = torch.cat([p1_1, p1_2], dim=3)
        x11 = torch.cat([p1, a1], dim=3)
        xa1, xp1 = self.conv_caps3_1(x11)
        a2 = torch.cat([a2_1, a2_2], dim=3)
        p2 = torch.cat([p2_1, p2_2], dim=3)
        x22 = torch.cat([p2, a2], dim=3)
        xa2, xp2 = self.conv_caps3_2(x22)
        xa = torch.cat([xa1, xa2], dim=3)
        xp = torch.cat([xp1, xp2], dim=3)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam13(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2. ConvCaps1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    3. ConvCaps2: D=32
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam13, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, 7, P, stride=1, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, 7, P, stride=1, iters=iters)
        self.conv_caps2 = ConvCaps(2*C, D, 1, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        a = torch.cat([a1, a2], dim=3)
        p = torch.cat([p1, p2], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps2(x)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose


class MyCapsNet_cam14(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2.1 ConvCaps1_1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
        ConvCaps2_1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    2.2 ConvCaps1_2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
        ConvCaps2_2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    3. ConvCaps3_1: D=32
        (_, 3, 3, 32x2x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
       ConvCaps3_2: D=32
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam14, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D, 7, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D, 7, P, stride=1, iters=iters)
        self.conv_caps3_1 = ConvCaps(D*2, D, K, P, stride=1, iters=iters)
        self.conv_caps3_2 = ConvCaps(D*2, D, 1, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D*2, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)
        self.class_caps1 = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)    #supervised for 3*3
        self.class_caps2 = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)    #supervised for 1*1

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1_1, p1_1 = self.conv_caps1_1(x1)    # 3*3
        a2_1, p2_1 = self.conv_caps2_1(x1)    # 1*1
        x2 = self.primary_caps2(x2)
        a1_2, p1_2 = self.conv_caps1_2(x2)
        a2_2, p2_2 = self.conv_caps2_2(x2)
        a1 = torch.cat([a1_1, a1_2], dim=3)
        p1 = torch.cat([p1_1, p1_2], dim=3)
        x11 = torch.cat([p1, a1], dim=3)
        xa1, xp1 = self.conv_caps3_1(x11)
        a2 = torch.cat([a2_1, a2_2], dim=3)
        p2 = torch.cat([p2_1, p2_2], dim=3)
        x22 = torch.cat([p2, a2], dim=3)
        xa2, xp2 = self.conv_caps3_2(x22)
        xa = torch.cat([xa1, xa2], dim=3)
        xp = torch.cat([xp1, xp2], dim=3)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        
        xa11 = torch.cat([xp1, xa1], dim=3)
        x_3, pose_3 = self.class_caps1(xa11)
        
        xa22 = torch.cat([xp2, xa2], dim=3)
        x_1, pose_1 = self.class_caps2(xa22)
        
        return x, pose, x_3, x_1


class MyCapsNet_cam15(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 512, 7, 7) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    2.1 ConvCaps1_1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
        ConvCaps2_1: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    2.2 ConvCaps1_2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
        ConvCaps2_2: C=32
        (_, 7, 7, 32x(4x4+1)) -> 7x7 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    3. ConvCaps3_1: D=32
        (_, 3, 3, 32x2x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
       ConvCaps3_2: D=32
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps1: E=2, for 3*3
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)
        ClassCaps2: E=2, for 1*1
        (_, 1, 1, 32x2x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)
        fusionfc: 
        (classcaps1 & classcaps1) -> 2 out fused class
        4 -> 2

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam15, self).__init__()
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D, 7, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D, 7, P, stride=1, iters=iters)
        self.conv_caps3_1 = ConvCaps(D*2, D, K, P, stride=1, iters=iters)
        self.conv_caps3_2 = ConvCaps(D*2, D, 1, P, stride=1, iters=iters)
        self.class_caps1 = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)    #supervised for 3*3
        self.class_caps2 = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)    #supervised for 1*1
        self.fusionfc = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.primary_caps1(x1)
        a1_1, p1_1 = self.conv_caps1_1(x1)    # 3*3
        a2_1, p2_1 = self.conv_caps2_1(x1)    # 1*1
        x2 = self.primary_caps2(x2)
        a1_2, p1_2 = self.conv_caps1_2(x2)
        a2_2, p2_2 = self.conv_caps2_2(x2)
        a1 = torch.cat([a1_1, a1_2], dim=3)
        p1 = torch.cat([p1_1, p1_2], dim=3)
        x11 = torch.cat([p1, a1], dim=3)
        xa1, xp1 = self.conv_caps3_1(x11)
        a2 = torch.cat([a2_1, a2_2], dim=3)
        p2 = torch.cat([p2_1, p2_2], dim=3)
        x22 = torch.cat([p2, a2], dim=3)
        xa2, xp2 = self.conv_caps3_2(x22)
        
        xa11 = torch.cat([xp1, xa1], dim=3)
        x_3, pose_3 = self.class_caps1(xa11)
        
        xa22 = torch.cat([xp2, xa2], dim=3)
        x_1, pose_1 = self.class_caps2(xa22)
        
        o = torch.cat((x_3, x_1), 1)
        o = self.fusionfc(o)
        o = self.softmax(o)
        
        return o, pose_3, x_3, x_1


        
class MyCapsNet_cam16(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 7x7x512(A=512), the feature maps change as follows:
    0. downsample feature map: A=512
        (_, 512, 7, 7) -> maxpool, 2x2 filter, stride 2
        x -> (_, 512, 3, 3)
    1. PrimaryCaps: B=32
        (_, 512, 3, 3) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 3, 3, 32x4x4), activation: (_, 3, 3, 32)
    2. ConvCaps1: C=32
        (_, 3, 3, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    3. ConvCaps2: D=32
        (_, 1, 1, 32x(4x4+1)) -> 1x1 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 1, 1, 32x4x4), activation: (_, 1, 1, 32)
    4. ClassCaps: E=2
        (_, 1, 1, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_cam16, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=1)
        self.primary_caps2 = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps1_2 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps2 = ConvCaps(2*C, D, 1, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x1, x2):
        x1 = self.pool1(x1)
        x1 = self.primary_caps1(x1)
        a1, p1 = self.conv_caps1_1(x1)
        x2 = self.pool2(x2)
        x2 = self.primary_caps2(x2)
        a2, p2 = self.conv_caps1_2(x2)
        a = torch.cat([a1, a2], dim=3)
        p = torch.cat([p1, p2], dim=3)
        x = torch.cat([p, a], dim=3)
        xa, xp = self.conv_caps2(x)
        x = torch.cat([xp, xa], dim=3)
        x, pose = self.class_caps(x)
        return x, pose
        
        
        
class MyCapsNet_bcnn(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 16x16x1024(A=1024), the feature maps change as follows:
    1. PrimaryCaps: B=32
        (_, 1024, 16, 16) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 16, 16, 32x4x4), activation: (_, 16, 16, 32)
    2. ConvCaps1: C=32
        (_, 16, 16, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 7, 7, 32x4x4), activation: (_, 7, 7, 32)
    3. ConvCaps2: D=32
        (_, 7, 7, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 5, 5, 32x4x4), activation: (_, 5, 5, 32)
    4. ClassCaps: E=2
        (_, 5, 5, 32x(4x4+1)) -> 1x1 conv, 2 out capsules
        x -> pose: (_, 2x4x4), activation: (_, 2)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(MyCapsNet_bcnn, self).__init__()
        self.se = SELayer(channel=512, reduction=4)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                        coor_add=False, w_shared=True)

    def forward(self, x):
        x1 = x
        x = self.se(x)
        bz, c, h, w = x.shape
        x = x.view(bz, c, h*w)
        x1 = x1.view(bz, c, h*w)
        x = (torch.bmm(x, torch.transpose(x1, 1, 2)) / (h*w)).view(bz, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = x.view(bz, 1024, 16, 16)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x, pose = self.class_caps(x)
        return x, pose



'''
TEST
Run this code with:
```
python -m capsules.py
```
'''
if __name__ == '__main__':
    model = capsules(E=10)
    print(model)
