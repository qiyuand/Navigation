# Relation Module
## Requirements
- Pytorch  0.4.1
- NumPy  1.11.3
- tqdm 4.26.0
- Tensorflow 1.12.0
- Tensorboard 1.12.0
- TensorboardX 1.4 (Enable Tensorboard in Pytorch)

## Description

- After installing all the dependencies above, run `generate_data.py` to generate a simple dataset, which will create two json files `./data/generated_data_train.json` and `./data/generated_data_test.json`.
- Run `main.sh` to start a training task using the dataset just generated. Tensorboard log files will be created under `./runs`. 

- **`model.py`** Relation Module are defined in this file.
- **`generate_data.py`**  Generate a simple dataset to debug by randomly sampling some objects in a scene. Run `generate_data.py` will create two json files `./data/generated_data_train.json` and `./data/generated_data_test.json`. There are 250000 data in the training set and 25000 data in the test set. For simplicity, the dataset will only contains two kind of relation: *OBJ_2 on the left of OBJ_1* and *OBJ_2 on the right of OBJ_1*. There are some paraphrases for each relation.
![demo-1](https://github.com/qiyuand/Relation-Module/blob/master/demo/demo1.png)
![demo-2](https://github.com/qiyuand/Relation-Module/blob/master/demo/demo2.png)

## Relation Module
- Basically, our architechture is the same as the paper [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf).
- The inputs to the module are the coordinates of each object in the scene, which are between 0 and 1, the category of each object, and the instruction. Before sending them into the model, we first map all the words in the category and instruction to their corresponding index in the `./data/vocabulary.json`.
> `output = self.RN(objs_coordinate, objs_category_idx, instruction_idx)`
- Inside the module, we use the same embedding layer to get the word vector for each word in the category and instruction. For simplicity, we did not use LSTM, and the embedding layer is randomly initialized. The maximum length of instruction is 10, and the embedding dimension is 128. We simply concatenate the embedding of every word to get the feature of the whole instruction, which should have the length of 128 * 10 = 1280.
- Then we did the same thing as the paper above.
![model](https://github.com/qiyuand/Relation-Module/blob/master/demo/model.png)

