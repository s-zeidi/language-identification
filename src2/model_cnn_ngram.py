import torch
import torch.nn as nn
import torch.nn.functional as F


class NgramCNN(nn.Module):

    def __init__(self, vocab_size, num_classes, embed_dim=128):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size+1, embed_dim)

        self.conv1 = nn.Conv1d(embed_dim, 256, 3)
        self.conv2 = nn.Conv1d(256, 256, 3)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(0,2,1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool(x).squeeze(-1)

        x = self.fc(x)

        return x