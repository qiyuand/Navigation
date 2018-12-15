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
