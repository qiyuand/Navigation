import sys

import re
import string
import json

import torch

import numpy as np
from PIL import Image

def get_all_scanId(path):
    scans = []
    with open(path,'r') as fp:
        for line in fp:
            if len(line) > 5:
                if '\n' in line:
                    scan = line.replace('\n','')
                else:
                    scan = line
                scans.append(scan)
    return scans

def sort_batch(instr_encoding):
    ''' Extract instructions from a list of observations and sort by descending
        sequence length (to enable PyTorch packing). '''
    base_vocab = ['<PAD>', '<UNK>', '<EOS>']
    padding_idx = base_vocab.index('<PAD>')

    seq_tensor = instr_encoding
    seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
    seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

    seq_lengths = torch.from_numpy(seq_lengths).long()

    # Sort sequences by lengths
    seq_lengths, perm_idx = seq_lengths.sort(0, True)

    # return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
    #        mask.byte().cuda(), \
    #        list(seq_lengths), list(perm_idx)
    return seq_lengths, perm_idx

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, vocabPath, encoding_length=10):
        self.encoding_length = encoding_length

        with open(vocabPath,'r') as fp:
            self.vocab = json.load(fp)
        self.word_to_index = {}

        # self.word_to_index['<UNK>'] = 0
        # self.word_to_index['<EOS>'] = 1
        # self.word_to_index['<PAD>'] = 2

        for i, word in enumerate(self.vocab):
            self.word_to_index[word] = i

    def build_vocab_from_dataset(self,dataset):
        self.vocab = []
        for i in dataset:
            self.vocab += i[2].split(' ')
        self.vocab = list(set(self.vocab))

        # save vocabulary into .txt file
        savePath = './vocabulary_from_dataset.json'
        with open(savePath,'w') as fp:
            json.dump(self.vocab,fp)


        for i, word in enumerate(self.vocab):
            self.word_to_index[word] = i

    def get_num_embedding(self):
        return len(self.vocab)

    def get_vocal_length(self):
        if self.vocab is None:
            return None
        else:
            return len(self.vocab)

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence,encoding_length=None):
        if encoding_length is None:
            encoding_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence[::1]):  # not reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:encoding_length]).reshape(1,-1)

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::1])  # not unreverse before output


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )