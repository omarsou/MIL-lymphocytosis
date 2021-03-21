from mil_framework.embedding_layer import SpecialEmbeddings
import torch.nn as nn
import torch


class LymphoModelClassifier(nn.Module):
    def __init__(self, features_embed, criterion, dim_embedding):
        super(LymphoModelClassifier, self).__init__()
        self.features_embed = features_embed
        self.fc0 = nn.Linear(dim_embedding, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.loss = criterion

    def forward(self, batch):
        # batch.features => (batch_size, N_images, 2560)
        features_embed = self.features_embed(batch.features, batch.features_mask)
        # features_embed => (batch_size, N_images, 1024)
        features_embed = features_embed.mean(dim=1)
        # features_embed => (batch_size, 1, 1024)
        features_embed = torch.cat([features_embed, batch.age, batch.concentration], dim=-1)
        # features_embed => (batch_size, 1, 1024 + 1 + 1) => (batch_size, 1, 1026)
        features_embed = self.activation(self.fc0(features_embed))
        # features_embed => (batch_size, 1, 512)
        features_embed = self.dropout(features_embed)
        features_embed = self.activation(self.fc1(features_embed))
        # features_embed => (batch_size, 1, 256)
        features_embed = self.dropout(features_embed)
        features_output = self.fc2(features_embed)
        # features_output => (batch_size, 1, 1)
        return features_output

    def get_metrics_for_batch(self, batch):
        output = self.forward(batch)
        loss_images = self.loss(output, batch.target.to(torch.float32))
        return loss_images, self.sigmoid(output)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


def build_model(cfg):
    features_embed = SpecialEmbeddings(embedding_dim=1024, input_size=2560, num_heads=cfg.num_heads, mask_on=False)
    criterion = nn.BCEWithLogitsLoss()
    model = LymphoModelClassifier(features_embed, criterion, 1026)
    model.apply(init_weights)
    if cfg.use_cuda:
        model.cuda()
    return model
