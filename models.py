import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import spectral_norm


class EmbedAll(nn.Module):
    def __init__(self, catlens=(3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3)):
        super(EmbedAll, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(catlen, min(50, catlen // 2)) for catlen in catlens]
        )

    def forward(self, labels):
        out = []
        for i, embedding in enumerate(self.embeddings):
            out.append(embedding(labels[:, i]))
        return torch.cat(out, dim=1)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, feature_dims=64, label_out_dims=98):
        super(Discriminator, self).__init__()
        fd = feature_dims
        lout = label_out_dims
        self.filtering = EmbedAll()
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(fd),
            nn.Mish(inplace=True))

        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(fd + lout, fd, 3, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(fd),
            nn.Mish(inplace=True),

            spectral_norm(nn.Conv2d(fd, 2 * fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(2 * fd),
            nn.Mish(inplace=True),

            spectral_norm(nn.Conv2d(2 * fd, 4 * fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(4 * fd),
            nn.Mish(inplace=True),

            spectral_norm(nn.Conv2d(4 * fd, 8 * fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(8 * fd),
            nn.Mish(inplace=True))

        self.final = nn.Sequential(
            nn.Conv2d(8 * fd, 1, 4, stride=1, padding=0, bias=False),
            nn.Flatten())

    def forward(self, image, label, return_features=False):
        label = self.filtering(label)
        label = label.unsqueeze(2).unsqueeze(3)
        label = label.repeat(1, 1, 32, 32)  # b, 64, 32, 32
        image = self.layer1(image)  # b,64,32,32
        image_and_label = torch.cat([image, label], dim=1)

        features = self.layers(image_and_label)
        validity = self.final(features)
        if return_features:
            return validity, features
        return validity


class Generator(nn.Module):
    def __init__(self, input_dims=256, feature_dims=64):
        super(Generator, self).__init__()
        fd = feature_dims
        self.filtering = EmbedAll()
        self.model = nn.Sequential(
            weight_norm(nn.ConvTranspose2d(input_dims, 8 * fd, 4, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(8 * fd),
            nn.Mish(inplace=True),

            weight_norm(nn.ConvTranspose2d(8 * fd, 4 * fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(4 * fd),
            nn.Mish(inplace=True),

            weight_norm(nn.ConvTranspose2d(4 * fd, 2 * fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(2 * fd),
            nn.Mish(inplace=True),

            weight_norm(nn.ConvTranspose2d(2 * fd, fd, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(fd),
            nn.Mish(inplace=True),

            weight_norm(nn.ConvTranspose2d(fd, 3, 4, stride=2, padding=1, bias=False)),
            nn.Tanh()
        )

    def forward(self, z, labels):
        batch_size = labels.size(0)
        filtered = self.filtering(labels)
        l_and_z = torch.cat([filtered, z], dim=1).view(batch_size, -1, 1, 1)
        out = self.model(l_and_z)
        return out
