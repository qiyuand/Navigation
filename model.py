import math

import torch
import torch.nn as nn
import torch.nn.functional as F

base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')


class FCOutputModel(nn.Module):
    def __init__(self, outFeatureChannel):
        super(FCOutputModel, self).__init__()

        self.fc3 = nn.Linear(outFeatureChannel, 4)

    def forward(self, x):
        x = F.dropout(x)
        x = self.fc3(x)
        return x


class RN(nn.Module):
    def __init__(self, num_embedding, embedding_dim, obj_num_max):
        super(RN, self).__init__()
        self.obj_num_max = obj_num_max

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)

        self.fc1 = nn.Linear(8 + 12 * embedding_dim, 1024)
        self.fc2 = nn.Linear(1024,1)
        self.fc3 = nn.Linear(obj_num_max ** 2,obj_num_max)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, objs_coordinate, objs_category_idx, instruction_idx):
        # 30 is the maximum object number

        bs = objs_coordinate.size(0)

        instr_embedding = self.embedding(instruction_idx)  # bs , 10 , embedding_dim
        # concatenate the embedding of each word
        instr_embedding = instr_embedding.view(bs, 1, -1)  # bs , 10 * embedding_dim
        instr_embedding = instr_embedding.expand(bs, 900, 10 * self.embedding_dim)

        category_embedding = self.embedding(objs_category_idx) # bs, 30, embedding_dim

        # concatenate the coordinates and category embedding of each object
        objs = torch.cat((objs_coordinate, category_embedding), dim=2)  # bs, 30, 4 + embedding_dim

        # generate 30 * 30 combination of every two objects
        objs_1 = objs.view(bs, 1, 30, 4 + self.embedding_dim)
        objs_1 = objs_1.expand(bs, 30, 30, 4 + self.embedding_dim)
        objs_1 = objs_1.contiguous().view(bs, 900, 4 + self.embedding_dim)

        objs_2 = objs.view(bs, 30, 1, 4 + self.embedding_dim)
        objs_2 = objs_2.expand(bs, 30, 30, 4 + self.embedding_dim)
        objs_2 = objs_2.contiguous().view(bs, 900, 4 + self.embedding_dim)

        objs_all_combination = torch.cat((objs_1, objs_2),dim=2) # bs, 900, 8 + 2 * embedding_dim

        # concatenate object feature with instruction embedding
        instr_embedding = instr_embedding
        x = torch.cat((objs_all_combination, instr_embedding), dim=2) # bs, 900, 8 + 12 * embedding_dim

        # go through each layer
        x = F.relu(self.fc1(x)) # bs, 900, 1024
        x = F.relu(self.fc2(x)) # bs, 900, 1
        x = x.view(bs, 900) # bs, 900
        x = F.softmax(self.fc3(x),dim=1) # bs, 30

        return x
