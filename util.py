import os
import shutil
import random

def select_and_copy_images(source='./data/test', target='./data/tmp', nums=15):
    os.makedirs(target, exist_ok=True)
    for dir in os.listdir(source):        
        source_path = os.path.join(source, dir)
        target_path = os.path.join(target, dir)
        if not os.path.isdir(source_path):
            continue
        os.makedirs(target_path, exist_ok=True)
        imgs = [f for f in os.listdir(source_path) if f != '.DS_Store']
        imgs = random.sample(imgs, min(nums, len(imgs)))
        for img in imgs:
            shutil.copy(os.path.join(source_path, img), os.path.join(target_path, img))


if __name__ == '__main__':
    select_and_copy_images()
