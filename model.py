import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1,
                                                                 keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class QModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        # Define image embedding
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        z = obs_space["image"][2]
        shape = n*m*z
        self.image_conv = nn.Sequential(
            nn.Linear(shape, shape//2),
            nn.ReLU(),
            nn.Linear(shape//2, shape//4),
            nn.ReLU(),
            nn.Linear(shape//4, 64),
            nn.ReLU()
        )
        self.embedding_size = 64

        self.word_embedding_size = 32
        self.word_embedding = nn.Embedding(obs_space["text"],
                                           self.word_embedding_size)
        self.text_embedding_size = 128
        self.text_rnn = nn.GRU(self.word_embedding_size,
                               self.text_embedding_size, batch_first=True)

        self.embedding_size += self.text_embedding_size
        # Define  model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.shape[0], -1)
        x = self.image_conv(x)

        embed_text = self._get_embed_text(obs.text)
        embedding = torch.cat((x, embed_text), dim=1)
        x = self.actor(embedding)

        return x

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
