import glob
import cv2
import argparse
import pandas as pd



def mask_to_bbox(mask_path):
    mask = cv2.imread(mask_path, 0)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min, x_max, y_max = 0, 0, 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = x if x < x_min or x_min == 0 else x_min
        y_min = y if y < y_min or y_min == 0 else y_min
        x_max = x + w if x + w > x_max or x_max == 0 else x_max
        y_max = y + h if y + h > y_max or y_max == 0 else y_max

    return [x_min, y_min, x_max, y_max]

def create_dataframe(bboxes, mask_paths):

    df = pd.DataFrame(columns=['image_id', 'x1', 'y1', 'x2', 'y2'])
    for i in range(len(mask_paths)):
        df.loc[i] = [mask_paths[i].split('/')[-1].split('.')[0]] + bboxes[i]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask-path', type=str, default='veriseti/masks', help='Path to mask images (default: \'veriseti/masks/*.png\')')
    parser.add_argument('--output-path', type=str, default='annotations.csv', help='Path to output csv file (default: \'annotations.csv\')')
    args = parser.parse_args()

    bboxes = []
    mask_paths = glob.glob(args.mask_path)
    for mask in mask_paths:
            bbox = mask_to_bbox(mask)
            bboxes.append(bbox)
    df = create_dataframe(bboxes, mask_paths)

    df.to_csv(args.output_path, index=False)
    
    print('Dataframe created and saved to {}'.format(args.output_path))
    
if __name__ == '__main__':
    main()

