import cv2
import os
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class BaseDataset(Dataset):
    """ 
    Args:
        dataset_path : dataset folder가 위치한 로컬 경로
        mode : train / val / test 중 선택
        transform : Compose된 transform
        global_config : 글로벌 설정 정보(config.json)
    """
    def __init__(self, dataset_path:str, mode:str, transform:A.Compose, global_config:dict):
        super().__init__()
        self.mode = mode
        self.dataset_path = dataset_path
        self.transform = transform
        self.global_config = global_config
        self.keys = list(self.global_config['Category'].keys())
        if self.mode == 'train':
            self.json_name = 'train_revised_final.json'
        elif mode == 'train_all':
            self.json_name = 'train_all.json'
        elif mode == 'val':
            self.json_name = 'val_revised_final.json'
        elif mode == 'test':
            self.json_name = 'test.json'
        else:
            raise ValueError("ERROR : Wrong mode")
        self.target_annotation_file = os.path.join(self.dataset_path, self.json_name)
        self.coco = COCO(self.target_annotation_file)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.dataset['images'][index]['id']#self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0] # type: ignore    
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # images /= 255.0
        
        if (self.mode in ('train', 'val', 'train_all')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True) # type: ignore    
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value =  self.keys.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())