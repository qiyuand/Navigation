import os
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader



class DatasetGenerator(Dataset):
    def __init__(self, datasetPath, vocabularyPath, objNumMax=30):
        self.objNumMax = objNumMax

        # load data from json file
        if os.path.exists(datasetPath):
            with open(datasetPath, 'r') as fp:
                self.data = json.load(fp)
        else:
            raise ValueError('No such dataset path')

        # load the vocabulary
        vocab_path = vocabularyPath
        with open(vocab_path,'r') as fp:
            self.vocab = json.load(fp)

        self.word_to_index = {}
        for i, word in enumerate(self.vocab):
            self.word_to_index[word] = i

        # 10 samples for sanity test
        # self.data = self.data[0:10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        objs = np.array(data['objs']).reshape(-1, 5)  # N * 5
        obj_num = objs.shape[0]
        obj_padding = self.objNumMax - obj_num

        # the coordinate of object
        objs_coordinate = objs[:,0:4].astype('float32')
        objs_coordinate = np.concatenate((objs_coordinate, np.zeros((obj_padding,4))),axis=0).astype('float32')

        # the category of object and its index in the vocabulary
        objs_category = objs[:,4].tolist()
        objs_category = objs_category + ['<PAD>'] * obj_padding
        objs_category_idx = np.array([self.word_to_index[i] for i in objs_category]).astype('int32')

        # the instruction of each data and the indexes of each word in the instruction
        instruction = data['instruction']
        instruction_idx = [self.word_to_index[i] for i in instruction.split()]
        if len(instruction_idx) < 10:
            instruction_idx += [self.word_to_index['<PAD>']] * (10 - len(instruction_idx))
        else:
            instruction_idx = instruction_idx[:10]
        instruction_idx = np.array(instruction_idx)

        # the target object index
        target = data['obj2_idx']

        return objs_coordinate, objs_category,objs_category_idx, instruction, instruction_idx, target, data


if __name__ == '__main__':
    path = '/Users/dongqiyuan/Desktop/boundingbox/v2/generated_data.json'
    datasetGenerator = DatasetGenerator(datasetPath=path)
    dataLoader = DataLoader(datasetGenerator, shuffle=True, batch_size=4,
                                          num_workers=12, pin_memory=False)
    for idx, (objs_coordinate, objs_category,objs_category_idx, instruction, instruction_idx, target) in enumerate(dataLoader):
        d = [objs_coordinate, objs_category,objs_category_idx, instruction, instruction_idx, target]