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
        return rows
    
    @staticmethod
    def replace(original_string):
        match_batch_num = re.search(r'BATCH (\d+)', original_string)
        match_format = re.search(r'\.(jpeg|jpg|png)$', original_string)
        replace = re.sub(r'BATCH \d+', f'BATCH {match_batch_num.group(1)} - ANNOTATED FILES', original_string)
        a_path = re.sub(match_format.group(0), 'A.csv', replace)
        b_path = re.sub(match_format.group(0), 'B.json', replace)
        return a_path, b_path 
    
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

    @staticmethod
    def visualize_bboxes(bboxes, labels, ax):
        """Draw bounding boxes with color-coded labels on a Matplotlib axis."""
        for box, label in zip(bboxes, labels):
            x_min, y_min, box_width, box_height = box
            # choose color based on label
            color = "red" if label == 'Cluster' or label == 'Clusters' else "blue"
            # choose linewidth
            linewidth = 3 if label == 'Cluster' or label == 'Clusters' else 0.2
            # draw bounding box
            ax.add_patch(plt.Rectangle(
                (x_min, y_min),
                box_width, box_height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none"
            ))
            # draw label text
            ax.text(
                x_min, y_min - 5,
                label,
                color=color,
                fontsize=2,
                weight="bold"
            )
    
    @staticmethod
    def image_tiling(original_image, tile_size=StaticVariable.tile_size, overlap=.25):
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
    
    def polygon_to_bounding_box(json_data, file):
        bounding_box_list = []
        for item in json_data[file].values():
            if isinstance(item, dict):
                if item != {}:
                    length = len(item.values())
                    for i in range(length):
                        poly_x = item[str(i)]['shape_attributes']['all_points_x']
                        poly_y = item[str(i)]['shape_attributes']['all_points_y']
                        # Compute bounding box
                        x_min, x_max = min(poly_x), max(poly_x)
                        y_min, y_max = min(poly_y), max(poly_y)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        # Match format: [x, y, width, height]
                        bounding_box_list.append([int(x_min), int(y_min), int(bbox_width), int(bbox_height)])
        return bounding_box_list
    
    def get_coordinates_intersections(x_min, t_x_min, y_min, t_y_min, x_max, t_x_max, y_max, t_y_max):
        return (
            max(x_min, t_x_min),   # overlap left
            max(y_min, t_y_min),   # overlap top
            min(x_max, t_x_max),   # overlap right
            min(y_max, t_y_max)    # overlap bottom
        )
            
    def get_normalize_bounding_box(x_min, y_min, bbox_width, bbox_height, img_width, img_height):
        x_center = (x_min + bbox_width / 2) / img_width
        y_center = (y_min + bbox_height / 2) / img_height
        w_norm = bbox_width / img_width
        h_norm = bbox_height / img_height
        return x_center, y_center, w_norm, h_norm
    
    def csv_data_to_annotations(csv_data):
        df_labels  = csv_data['label_name'].tolist()  # Assuming single class for simplicity
        df_bboxes = csv_data[['bbox_x','bbox_y','bbox_width','bbox_height']].apply(lambda x: [x['bbox_x'], x['bbox_y'], x['bbox_width'], x['bbox_height']], axis=1).tolist()
        return df_labels, df_bboxes
    
    def json_data_to_annotations(json_data, file):
        cluster_bboxes = Utils.polygon_to_bounding_box(json_data, file)
        cluster_labels = ["Cluster"] * len(cluster_bboxes)
        return cluster_labels, cluster_bboxes
        
    def adjust_bboxes_for_tile(annotations, x0, y0,
                               tile_size=StaticVariable.tile_size,
                               min_pixel_size=StaticVariable.min_pixel_size):
        tile_bboxes = []
        tile_labels = []
        for annotation in annotations:
            label, bbox = annotation
            x_min, y_min, bbox_width, bbox_height = bbox
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height
            
            t_x_min, t_y_min = x0, y0
            t_x_max, t_y_max = x0 + tile_size, y0 + tile_size
            
            ix1, iy1, ix2, iy2 = Utils.get_coordinates_intersections(
                x_min, t_x_min, y_min, t_y_min, x_max, t_x_max, y_max, t_y_max
            )

            if ix1 < ix2 and iy1 < iy2:
                new_width = ix2 - ix1
                new_height = iy2 - iy1
                if new_width >= min_pixel_size and new_height >= min_pixel_size:
                    new_x = ix1 - x0
                    new_y = iy1 - y0
                    tile_bboxes.append((new_x, new_y, new_width, new_height))
                    tile_labels.append(label)
        return tile_labels, tile_bboxes
                    # yield label, (new_x, new_y, new_width, new_height)
    
    def get_bboxes_and_labels(image_path, file):
        csv_path, json_path  = Utils.replace(image_path)
        csv_data = Utils.get_csv_data(csv_path)
        json_data = Utils.get_json_data(json_path)
        df_labels, df_bboxes = Utils.csv_data_to_annotations(csv_data)
        cluster_labels, cluster_bboxes = Utils.json_data_to_annotations(json_data, file)
        return df_labels, df_bboxes, cluster_labels, cluster_bboxes
        
    @staticmethod
    def preprocess(file_path, invalid):
        for root, _, files in os.walk(file_path):
            for file in files:
                format = os.path.splitext(file)[1]
                if StaticVariable.is_supported(format):
                    image_path = os.path.join(root, file)
                    if image_path not in invalid:
                        
                        df_labels, df_bboxes, cluster_labels, cluster_bboxes = Utils.get_bboxes_and_labels(image_path, file)
                        
                        # Combine both
                        all_bboxes = df_bboxes + cluster_bboxes
                        all_labels = df_labels + cluster_labels
                        rgb_image, img_height, img_width = Utils.get_image_data(image_path)
                        
                        annotations = list(zip(all_labels, all_bboxes))
                        
                        print(f'Processing {file}...')
                        for tile, x0, y0, tile_id in Utils.image_tiling(rgb_image):
                            tile_labels, tile_bboxes = Utils.adjust_bboxes_for_tile(annotations, x0, y0)
                            # for label, (new_x, new_y, new_width, new_height) in Utils.adjust_bboxes_for_tile(annotations, x0, y0):
                            #     print(f'Tile ID: {tile_id}, Label: {label}, BBox: ({new_x}, {new_y}, {new_width}, {new_height})')
                            fig, ax = plt.subplots(1, figsize=(5, 5))
                            ax.imshow(tile)
                            Utils.visualize_bboxes(tile_bboxes, tile_labels, ax)
                            plt.title(tile_id)
                            plt.show()
                            break
                break
                    
if __name__ == '__main__':
    invalid = Utils.check_dataset(StaticVariable.data_path)
    Utils.preprocess(StaticVariable.data_path, invalid)