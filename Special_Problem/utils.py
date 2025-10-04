import os
from static_variables import StaticVariable
import re
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt

class Utils:
    @staticmethod
    def check_dataset(file_path):
        rows = []
        for root, _, files in os.walk(file_path):
            for file in files:
                format = os.path.splitext(file)[1]
                if StaticVariable.is_supported(format):
                    image_path = os.path.join(root, file)
                    a_path, b_path  = Utils.replace(image_path)
                    if not os.path.exists(a_path) or not os.path.exists(b_path):
                        rows.append(image_path)
        rows = pd.DataFrame(rows, columns=['Annotation Problems'])
        rows.to_csv('annotation_problems.csv', index=False)
    
    @staticmethod
    def replace(original_string):
        match_batch_num = re.search(r'BATCH (\d+)', original_string)
        match_format = re.search(r'\.(jpeg|jpg|png)$', original_string)
        replace = re.sub(r'BATCH \d+', f'BATCH {match_batch_num.group(1)} - ANNOTATED FILES', original_string)
        a_path = re.sub(match_format.group(0), 'A.csv', replace)
        b_path = re.sub(match_format.group(0), 'B.json', replace)
        return a_path, b_path 
    
    @staticmethod
    def preprocess(file_path):
        for root, _, files in os.walk(file_path):
            for file in files:
                format = os.path.splitext(file)[1]
                if StaticVariable.is_supported(format):
                    image_path = os.path.join(root, file)
                    a_path, b_path  = Utils.replace(image_path)
    
    @staticmethod
    def get_json_data(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def get_csv_data(csv_path):
        data = pd.read_csv(csv_path)
        return data
    
    @staticmethod
    def get_image_data(image_path):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ =  rgb_image.shape
        return rgb_image, img_height, img_width
    
    @staticmethod
    def write_normalize_bounding_box(
        x_min, y_min, bbox_width, bbox_height, img_width, img_height,
        output_path, class_id):
        x_center, y_center, w_norm, h_norm = Utils.get_normalize_bounding_box(
            x_min, y_min, bbox_width, bbox_height, img_width, img_height
            )
        with open(output_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    def get_normalize_bounding_box(x_min, y_min, bbox_width, bbox_height, img_width, img_height):
        x_center = (x_min + bbox_width / 2) / img_width
        y_center = (y_min + bbox_height / 2) / img_height
        w_norm = bbox_width / img_width
        h_norm = bbox_height / img_height
        return x_center, y_center, w_norm, h_norm
    
    @staticmethod
    def visualize_bboxes(bboxes, labels, ax):
        """Draw bounding boxes with color-coded labels on a Matplotlib axis."""
        for box, label in zip(bboxes, labels):
            x_min, y_min, box_width, box_height = box
            # choose color based on label
            classes = {0: "Cluster", 1: "Thyrocyte"}
            color = "blue" if label == 0 else "red"
            # draw bounding box
            ax.add_patch(plt.Rectangle(
                (x_min, y_min),
                box_width, box_height,
                linewidth=0.5,
                edgecolor=color,
                facecolor="none"
            ))
            # draw label text
            ax.text(
                x_min, y_min - 5,
                classes.get(label, str(label)),
                color=color,
                fontsize=5,
                weight="bold"
            )
        plt.show()
    
    @staticmethod
    def image_tiling(original_image, tile_size=512, overlap=.25):
        """
        Generate tiles from the original image with specified overlap.
        Yields tuples of (tile, x0, y0, tile_id) where (x0, y0) is the top-left
        coordinate of the tile in the original image and tile_id is a unique identifier.
        """
        stride = int(tile_size * (1 - overlap))
        img_height, img_width, _ = original_image.shape
        for row_idx, y0 in enumerate(range(0, img_height, stride)):
            for col_idx, x0 in enumerate(range(0, img_width, stride)):
                tile_id = f"{row_idx}_{col_idx}"
                x1 = min(x0 + tile_size, img_width)
                y1 = min(y0 + tile_size, img_height)
                tile = original_image[y0:y1, x0:x1]
                yield tile, x0, y0, tile_id
                
if __name__ == '__main__':
    Utils.preprocess(StaticVariable.data_path)