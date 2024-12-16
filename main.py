import cv2
import os
import json
from tqdm import tqdm
import numpy as np

# 更改变量名为更具描述性的名称
province_codes = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabet_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ad_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 更改函数名为更具描述性的名称
def generate_labels(image_directory, ground_truth_directory, dataset_phase):
    cropped_image_save_dir = os.path.join(ground_truth_directory, dataset_phase, 'cropped_images')
    os.makedirs(cropped_image_save_dir, exist_ok=True)

    detection_file = open(os.path.join(ground_truth_directory, dataset_phase, 'detection.txt'), 'w', encoding='utf-8')
    recognition_file = open(os.path.join(ground_truth_directory, dataset_phase, 'recognition.txt'), 'w', encoding='utf-8')

    image_counter = 0
    for filename in tqdm(os.listdir(os.path.join(image_directory, dataset_phase))):
        file_parts = filename.split('-')
        if len(file_parts) < 5:
            continue
        coordinates = file_parts[3].split('_')
        text_parts = file_parts[4].split('_')
        bounding_boxes = []
        for coord in coordinates:
            bounding_boxes.append([int(x) for x in coord.split("&")])
        bounding_boxes = [bounding_boxes[2], bounding_boxes[3], bounding_boxes[0], bounding_boxes[1]]
        license_plate_number = province_codes[int(text_parts[0])] + alphabet_codes[int(text_parts[1])] + ''.join([ad_codes[int(x)] for x in text_parts[2:]])

        # Detection information
        detection_info = [{'coordinates': bounding_boxes, 'transcription': license_plate_number}]
        detection_file.write('{}\t{}\n'.format(os.path.join(dataset_phase, filename), json.dumps(detection_info, ensure_ascii=False)))

        # Recognition information
        bounding_boxes = np.float32(bounding_boxes)
        image = cv2.imread(os.path.join(image_directory, dataset_phase, filename))
        cropped_image = get_cropped_image(image, bounding_boxes)
        cropped_image_filename = '{}_{}.jpg'.format(image_counter, '_'.join(text_parts))
        cropped_image_path = os.path.join(cropped_image_save_dir, cropped_image_filename)
        cv2.imwrite(cropped_image_path, cropped_image)
        recognition_file.write('{}/cropped_images/{}\t{}\n'.format(dataset_phase, cropped_image_filename, license_plate_number))
        image_counter += 1
    detection_file.close()
    recognition_file.close()

def get_cropped_image(image, points):
    assert len(points) == 4, "points array must contain exactly 4 coordinates"
    crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    standard_points = np.float32([[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]])
    transformation_matrix = cv2.getPerspectiveTransform(points, standard_points)
    cropped_image = cv2.warpPerspective(image, transformation_matrix, (crop_width, crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    height, width = cropped_image.shape[0:2]
    if height * 1.0 / width

        img_dir = 'D:\\ccpd\\CCPD2020\\ccpd_green'
        save_gt_folder = 'D:\\ccpd\\CCPD2020'
        # phase = 'train' # change to val and test to make val dataset and test dataset
        for phase in ['train', 'val', 'test']:
            make_label(img_dir, save_gt_folder, phase)
