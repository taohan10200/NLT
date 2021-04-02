import os
import random
from shutil import copyfile
source_root = '/media/D/ht/ProcessedData/txt_list/'


def GCC_scene_divide():
    with open(os.path.join(source_root+'all_list.txt')) as f:
        lines = f.readlines()
        for line in lines:
            scene_id = line.split('_')[3]
            with open(os.path.join(source_root, 'GCC_scenes', scene_id + '.txt'),'a') as f:
                f.write(line)
                f.close()
            print(scene_id)


def train_val_test_divide():
    root = '/media/D/ht/ProcessedData/txt_list/GCC_scenes'
    all_scenes = os.listdir(os.path.join(root))
    number = len(all_scenes)
    print("all_train_scenes:", number)

    val_scenes = random.sample(all_scenes, 16)

    all_scenes = set(all_scenes)
    val_scenes = set(val_scenes)
    train_test = all_scenes - val_scenes

    test_scenes = random.sample(train_test, 20)
    test_scenes = set(test_scenes)

    train_scenes = train_test - test_scenes

    print("all_train_number:", len(train_scenes))
    print("all_validation_number:", len(val_scenes))
    print("all_train_number:", len(test_scenes))

    des = './data/GCC_scenes'
    os.makedirs(os.path.join(des,'train'),mode=0o777,exist_ok=True)
    os.makedirs(os.path.join(des, 'val'), mode=0o777,exist_ok=True)
    os.makedirs(os.path.join(des, 'test'), mode=0o777,exist_ok=True)
    print(test_scenes)
    for scene in train_scenes:

        copyfile(os.path.join(root,scene),os.path.join(des,'train',scene))
    for scene in val_scenes:
        copyfile(os.path.join(root,scene),os.path.join(des,'val',scene))
    for scene in test_scenes:
        copyfile(os.path.join(root,scene),os.path.join(des,'test',scene))

