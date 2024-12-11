import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def denorm(image, stats):
    if image.shape[0] > 1 and not isinstance(stats[0], (float, int)):
        for i in range(len(image)):
            image[i] = image[i] * stats[1][i] + stats[0][i]
    else:
        image = image * stats[1] + stats[0]
    return image


def load_generator():
    return torch.load("Models/generator.pt", map_location=device)


class ImageGenerator:
    def __init__(self, generator):
        self.device = device
        self.gen = generator.to(self.device)
        self.gen.eval()
        self.catlens = (3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3)
        self.latent_dims = 158

    def get_noise(self, noise_dims=158):
        return torch.randn(1, noise_dims, device=self.device)

    def check_labels(self, labels):
        for i, (label, catlen) in enumerate(zip(labels, self.catlens)):
            if (label >= catlen) or (label < 0) or (not isinstance(label, int)):
                raise ValueError(f"Either label is exceeding maximum value {catlen - 1}, or incorrect data is passed!")
        return True

    @torch.no_grad()
    def generate(self, labels):
        if isinstance(labels, dict):
            labels = list(labels.values())
        self.gen.eval()
        if self.check_labels(labels):
            noise = self.get_noise()
            labels = torch.tensor(labels).unsqueeze(0).to(self.device)
            image = denorm(self.gen(noise, labels), (0.5, 0.5)).squeeze(0).permute(1, 2, 0).cpu()
            return (image.numpy() * 255).astype(np.uint8)
