import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):

    def __init__(self, vocab_size, num_classes):

        super().__init__()

        embed_dim = 256

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        x = self.dropout(x)

        x = self.fc(x)

        return x