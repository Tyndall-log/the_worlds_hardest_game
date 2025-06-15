import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import create_mlp

class ImageOnlyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)

        # CNN for image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # assume (3, 84, 84)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute flattened CNN output
        with th.no_grad():
            sample_input = th.as_tensor(observation_space["image"].sample()[None].transpose(0, 3, 1, 2)).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU()
        )

        self._features_dim = 256

    def forward(self, observations):
        # image: (B, H, W, C) → (B, C, H, W)
        x = observations["image"].permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.cnn(x))