
import json

import numpy as np
import matplotlib.pyplot as plt
import tqdm

def get_mpcat40(path='../data/mpcat40.txt'):
    mpcat40 = []
    with open(path, 'r') as fp:
        for line in fp:
            mpcat40.append(line.split()[1])
    mpcat40.remove('void')
    mpcat40.remove('unlabeled')
    mpcat40.remove('objects')
    mpcat40.remove('misc')

    return mpcat40


def sample_category():
    return np.random.choice(mpcat40, 1)[0]


def sample_new_obj(length_mu=0.1, length_sigma=0.05):
    x_bottom_left, y_bottom_left, x_top_right, y_top_right = 99, 99, 99, 99

    while np.min([x_bottom_left, y_bottom_left, x_top_right, y_top_right]) <= 0.0 or np.max(
            [x_bottom_left, y_bottom_left, x_top_right, y_top_right]) >= 1.0:
        x_bottom_left, y_bottom_left = np.random.uniform(0.05, 0.95, 2)

        x_length, y_length = -1, -1
        while x_length <= 0.0 or y_length <= 0.0:
            x_length, y_length = np.random.normal(length_mu, length_sigma, 2)
        x_top_right, y_top_right = x_bottom_left + x_length, y_bottom_left + y_length

    return [x_bottom_left, y_bottom_left, x_top_right, y_top_right]


def overlap_test(obj1, obj2):
    if np.min([obj1[2], obj2[2]]) >= np.max([obj1[0], obj2[0]]) and np.min([obj1[3], obj2[3]]) >= np.max(
            [obj1[1], obj2[1]]):
        return False
    else:
        return True


def generate_scene(obj_num, length_mu=0.1, length_sigma=0.05):
    objs = []
    new_obj = sample_new_obj(length_mu, length_sigma)
    new_obj.append(sample_category())
    objs.append(new_obj)

    for i in range(obj_num - 1):
        overlap = True
        while overlap:
            overlap = False
            new_obj = sample_new_obj(length_mu, length_sigma)
            for o in objs:
                if not overlap_test(o, new_obj):
                    overlap = True
                    break
        new_obj.append(sample_category())
        objs.append(new_obj)

    return objs


def draw_scene(objs, special_idx=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for idx, o in enumerate(objs):
        x_len = o[2] - o[0]
        y_len = o[3] - o[1]
        x = o[0]
        y = o[1]
        category = o[4]
        if special_idx and idx == special_idx:
            rect = plt.Rectangle((x, y), x_len, y_len, color='coral')
        else:
            rect = plt.Rectangle((x, y), x_len, y_len)
        if title:
            plt.title(title)
        plt.text(x, y, category)
        ax.add_patch(rect)

    plt.show()


def OBJ_LEFT_OF_OBJ(objs, x_threshold=0.4, y_threshold=0.2):
    patterns = [
        'the %s on the left of %s',
        'the %s to the left of %s',
        'the %s on the left side of %s',
        'the %s to the left side of %s'
    ]
    objs_num = len(objs)
    coor = [i[0:4] for i in objs]
    cate = [i[4] for i in objs]
    coor_np = np.array(coor)
    cate_np = np.array(cate)
    # if obj1 has no objects on its left, choose another one
    resample = True
    while resample:
        resample = False
        obj1_idx = np.random.choice(len(objs), 1)[0]
        obj1 = objs[obj1_idx]
        obj1_coor = coor[obj1_idx]

        # objs on its left
        flag1 = coor_np[:, 2] < obj1_coor[0]

        # objs whose y_center within threshold
        x_center = (coor_np[:, 0] + coor_np[:, 2]) / 2.0
        y_center = (coor_np[:, 1] + coor_np[:, 3]) / 2.0
        obj1_x_center = (obj1_coor[0] + obj1_coor[2]) / 2.0
        obj1_y_center = (obj1_coor[1] + obj1_coor[3]) / 2.0
        flag2 = np.abs(x_center - obj1_x_center) < x_threshold
        flag3 = np.abs(y_center - obj1_y_center) < y_threshold

        flag = (flag1 * flag2 * flag3)
        candidates = np.arange(objs_num)[flag]
        if np.all(flag == False):
            resample = True

    # choose the nearest, possibility is the distance
    distance = []
    for idx in candidates:
        dis = np.sqrt((x_center[idx] - obj1_x_center) ** 2 + (y_center[idx] - obj1_y_center) ** 2)
        distance.append(dis)
    distance = np.array(distance)
    p = distance / np.sum(distance)

    obj2_idx = np.random.choice(candidates, 1, False, p)[0]
    obj2 = objs[obj2_idx]

    ptn_idx = np.random.choice(len(patterns), 1)[0]

    instruction = patterns[ptn_idx] % (cate[obj2_idx], cate[obj1_idx])

    osv = ((cate_np == cate[obj1_idx]) + (cate_np == cate[obj2_idx]))
    osv = np.arange(objs_num)[osv]

    data = {'objs': objs,
            'instruction': instruction,
            'obj12': [obj1, obj2],
            'obj1_idx': obj1_idx,
            'obj2_idx': obj2_idx,
            'osv': osv.tolist()}

    return data

def OBJ_RIGHT_OF_OBJ(objs, x_threshold=0.4, y_threshold=0.2):
    patterns = [
        'the %s on the right of %s',
        'the %s to the right of %s',
        'the %s on the right side of %s',
        'the %s to the right side of %s'
    ]
    objs_num = len(objs)
    coor = [i[0:4] for i in objs]
    cate = [i[4] for i in objs]
    coor_np = np.array(coor)
    cate_np = np.array(cate)
    # if obj1 has no objects on its left, choose another one
    resample = True
    while resample:
        resample = False
        obj1_idx = np.random.choice(len(objs), 1)[0]
        obj1 = objs[obj1_idx]
        obj1_coor = coor[obj1_idx]

        # objs on its right
        flag1 = coor_np[:, 0] > obj1_coor[2]

        # objs whose y_center within threshold
        x_center = (coor_np[:, 0] + coor_np[:, 2]) / 2.0
        y_center = (coor_np[:, 1] + coor_np[:, 3]) / 2.0
        obj1_x_center = (obj1_coor[0] + obj1_coor[2]) / 2.0
        obj1_y_center = (obj1_coor[1] + obj1_coor[3]) / 2.0
        flag2 = np.abs(x_center - obj1_x_center) < x_threshold
        flag3 = np.abs(y_center - obj1_y_center) < y_threshold

        flag = (flag1 * flag2 * flag3)
        candidates = np.arange(objs_num)[flag]
        if np.all(flag == False):
            resample = True

    # choose the nearest, possibility is the distance
    distance = []
    for idx in candidates:
        dis = np.sqrt((x_center[idx] - obj1_x_center) ** 2 + (y_center[idx] - obj1_y_center) ** 2)
        distance.append(dis)
    distance = np.array(distance)
    p = distance / np.sum(distance)

    obj2_idx = np.random.choice(candidates, 1, False, p)[0]
    obj2 = objs[obj2_idx]

    ptn_idx = np.random.choice(len(patterns), 1)[0]

    instruction = patterns[ptn_idx] % (cate[obj2_idx], cate[obj1_idx])

    osv = ((cate_np == cate[obj1_idx]) + (cate_np == cate[obj2_idx]))
    osv = np.arange(objs_num)[osv]

    data = {'objs': objs,
            'instruction': instruction,
            'obj12': [obj1, obj2],
            'obj1_idx': obj1_idx,
            'obj2_idx': obj2_idx,
            'osv': osv.tolist()}

    return data

if __name__ == '__main__':
    mpcat40 = get_mpcat40()

    # Training Data
    # the amount of training data is NUM_SCENE * NUM_DATA_PER_SCENE
    data = []
    NUM_SCENE = range(10000)
    OBJ_MIN = 10
    OBJ_MAX = 30
    NUM_DATA_PER_SCENE = 25

    bar = tqdm.tqdm(NUM_SCENE)

    for i in bar:
        num_obj = int(np.random.uniform(OBJ_MIN,OBJ_MAX,1)[0])
        scene = generate_scene(num_obj)
        for j in range(NUM_DATA_PER_SCENE):
            data.append(OBJ_LEFT_OF_OBJ(scene))
        for j in range(NUM_DATA_PER_SCENE):
            data.append(OBJ_RIGHT_OF_OBJ(scene))

    data_save_path = './data/generated_data_train.json'
    with open(data_save_path,'w') as fp:
        json.dump(data,fp,indent=4)

    # Test Data
    # the amount of test data is NUM_SCENE * NUM_DATA_PER_SCENE
    data = []
    NUM_SCENE = range(1000)
    OBJ_MIN = 10
    OBJ_MAX = 30
    NUM_DATA_PER_SCENE = 25

    bar = tqdm.tqdm(NUM_SCENE)

    for i in bar:
        num_obj = int(np.random.uniform(OBJ_MIN,OBJ_MAX,1)[0])
        scene = generate_scene(num_obj)
        for j in range(NUM_DATA_PER_SCENE):
            data.append(OBJ_LEFT_OF_OBJ(scene))
        for j in range(NUM_DATA_PER_SCENE):
            data.append(OBJ_RIGHT_OF_OBJ(scene))

    data_save_path = './data/generated_data_test.json'
    with open(data_save_path,'w') as fp:
        json.dump(data,fp,indent=4)