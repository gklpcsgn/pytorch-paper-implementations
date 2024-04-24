import cv2
import numpy as np
import torch
import glob


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, bboxes,mask_paths):
        self.img_paths = glob.glob(img_paths)
        self.mask_paths = glob.glob(mask_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = cv2.imread(img_path)
        
        img_id = img_path.split('/')[-1].split('.')[0]

        # Resize image to 572x572
        img = cv2.resize(img, (572, 572))

        # Normalize image
        img = img / 255.0

        # convert to tensor
        img = np.array(img)

        # change channels to first dimension as expected by pytorch [C, H, W]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # convert to 0-1 mask
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        mask = torch.from_numpy(mask).float()

        return img, mask

    def __len__(self):
        return len(self.img_paths)