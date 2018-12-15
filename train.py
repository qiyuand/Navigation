import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

from model import RN
from dataset import DatasetGenerator
from utils import Tokenizer


class Task():
    def __init__(self, args):
        print '#' * 60
        print ' ' * 20 + '    Task Created    ' + ' ' * 20
        print '#' * 60

        ######################################################################################################
        # Parameters
        self.batchSize = args.batchSize
        self.lr = args.lr
        self.weightDecay = 1e-4

        self.objNumMax = 30
        self.wordEmbeddingDim = 128
        self.instructionLength = 10
        self.pinMemory = True
        self.dropout = False

        self.epoch = args.epoch
        self.epoch_i = 0

        self.batchPrint = 100
        self.batchModelSave = args.batchModelSave
        self.checkPoint = args.checkPoint

        # Path
        self.vocabularyPath = './data/vocabulary.json'
        self.trainDatasetPath = './data/generated_data_train.json'
        self.testDatasetPath = './data/generated_data_test.json'
        self.logPath = args.logPath

        # Dataset
        self.trainDataset = DatasetGenerator(datasetPath=self.trainDatasetPath, vocabularyPath=self.vocabularyPath)
        self.testDataset = DatasetGenerator(datasetPath=self.testDatasetPath, vocabularyPath=self.vocabularyPath)

        # Tokenizer
        self.tokenizer = Tokenizer(vocabPath=self.vocabularyPath)
        self.num_embedding = self.tokenizer.get_num_embedding()

        # DataLoader
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, shuffle=True, batch_size=self.batchSize,
                                          num_workers=12, pin_memory=self.pinMemory)
        self.testDataLoader = DataLoader(dataset=self.testDataset, shuffle=False, batch_size=self.batchSize,
                                         num_workers=12, pin_memory=self.pinMemory)
        # calculate batch numbers
        self.trainBatchNum = int(np.ceil(len(self.trainDataset) / float(self.batchSize)))
        self.testBatchNum = int(np.ceil(len(self.testDataset) / float(self.batchSize)))

        # Create model
        self.RN = RN(num_embedding=self.num_embedding, embedding_dim=self.wordEmbeddingDim, obj_num_max=self.objNumMax)

        # Run task on all available GPUs
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("Use ", torch.cuda.device_count(), " GPUs")
                self.RN = nn.DataParallel(self.RN)
            self.RN = self.RN.cuda()
            print 'Model Created on GPUs.'

        # Optermizer
        self.optimizer = optim.Adam(self.RN.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=self.weightDecay)
        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, mode='min')

        # Loss Function
        self.loss = torch.nn.CrossEntropyLoss()

        # Load model if a checkPoint is given
        if self.checkPoint != "":
            self.load(self.checkPoint)

        # TensorboardX record
        self.writer = SummaryWriter()
        self.stepCnt_train = 1
        self.stepCnt_test = 1

    def train(self):

        print 'Training task begin.'
        print '----Batch Size: %d' % self.batchSize
        print '----Learning Rate: %f' % (self.lr)
        print '----Epoch: %d' % self.epoch
        print '----Log Path: %s' % self.logPath

        for self.epoch_i in range(self.epoch):
            self.epochTrain()
            self.test()

    def epochTrain(self):
        s = '#' * 30 + '    Epoch %3d / %3d    ' % (self.epoch_i + 1, self.epoch) + '#' * 30
        print s
        bar = tqdm(self.trainDataLoader)
        for idx, (
        objs_coordinate, objs_category, objs_category_idx, instruction, instruction_idx, target, data) in enumerate(
                bar):

            batchSize = objs_coordinate.shape[0]

            # to cuda
            if torch.cuda.is_available():
                objs_coordinate = objs_coordinate.cuda()
                objs_category_idx = objs_category_idx.long().cuda()
                instruction_idx = instruction_idx.long().cuda()
                target = target.cuda()

            # Go through the model
            output = self.RN(objs_coordinate, objs_category_idx, instruction_idx)

            # calculate loss
            lossValue = self.loss(input=output, target=target)

            # Tensorboard record
            self.writer.add_scalar('Loss/Train', lossValue.item(), self.stepCnt_train)

            # print loss
            bar.set_description('Epoch: %d    Loss: %f' % (self.epoch_i + 1, lossValue.item()))

            # Backward
            self.optimizer.zero_grad()
            lossValue.backward()
            self.optimizer.step()
            # self.scheduler.step(lossValue)

            # Save model
            if (idx + 1) % self.batchModelSave == 0:
                self.save(batchIdx=(idx + 1))

            if idx % self.batchPrint == 0:
                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                s = ''
                for batch_i in range(batchSize):
                    s += 'Target: '
                    s += str(target[batch_i])
                    s += ' Output: '
                    s += ', '.join([str(i) for i in output[batch_i, :].tolist()])
                    s += '    #########    '
                self.writer.add_text('Target & Output', s, self.stepCnt_train)

                self.writer.add_histogram('output', output, self.stepCnt_train)
                self.writer.add_histogram('target', target, self.stepCnt_train)

                for name, param in self.RN.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.stepCnt_train)

            self.stepCnt_train += 1
            del lossValue

    def test(self):
        s = '#' * 28 + '  Test  Epoch %3d / %3d    ' % (self.epoch_i + 1, self.epoch) + '#' * 28
        print s

        bar = tqdm(self.testDataLoader)
        for idx, (
        objs_coordinate, objs_category, objs_category_idx, instruction, instruction_idx, target, data) in enumerate(
                bar):

            batchSize = objs_coordinate.shape[0]

            # to cuda
            if torch.cuda.is_available():
                objs_coordinate = objs_coordinate.cuda()
                objs_category_idx = objs_category_idx.cuda()
                instruction_idx = instruction_idx.cuda()
                target = target.cuda()

            # Go through the model
            output = self.RN(objs_coordinate, objs_category_idx, instruction_idx)

            # calculate loss
            lossValue = self.loss(input=output, target=target)
            # Tensorboard record
            self.writer.add_scalar('Loss/Test', lossValue.item(), self.stepCnt_test)

            # print loss
            bar.set_description('Epoch: %d    Loss: %f' % (self.epoch_i + 1, lossValue.item()))

            # Save model
            if (idx + 1) % self.batchModelSave == 0:
                self.save(batchIdx=(idx + 1))

            if idx % self.batchPrint == 0:
                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                s = ''
                for batch_i in range(batchSize):
                    s += 'Target: '
                    s += str(target[batch_i])
                    s += ' Output: '
                    s += ', '.join([str(i) for i in output[batch_i, :].tolist()])
                    s += '    #########    '
                self.writer.add_text('Target & Output Test', s, self.stepCnt_test)

                self.writer.add_histogram('output', output, self.stepCnt_test)
                self.writer.add_histogram('target', target, self.stepCnt_test)

                # for name, param in self.RN.named_parameters():
                #     self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.stepCnt_test)

            self.stepCnt_test += 1
            del lossValue

    def save(self, batchIdx=None):
        dirPath = os.path.join(self.logPath, 'models')

        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        if batchIdx is None:
            path = os.path.join(dirPath, 'Epoch-%03d-end.pth.tar' % (self.epoch_i + 1))
        else:
            path = os.path.join(dirPath, 'Epoch-%03d-Batch-%04d.pth.tar' % (self.epoch_i + 1, batchIdx))

        torch.save({'epochs': self.epoch_i + 1,
                    'batch_size': self.batchSize,
                    'lr': self.lr,
                    'weight_dacay': self.weightDecay,
                    'RN_model_state_dict': self.RN.state_dict()},
                   path)
        print 'Training log saved to %s' % path

    def load(self, path):
        modelCheckpoint = torch.load(path)
        self.RN.load_state_dict(modelCheckpoint['RN_model_state_dict'])
        print 'Load model from %s' % path
