import albumentations as A
import pandas as pd

class StaticVariable:
    formats = ['.jpeg', '.jpg', '.png']
    data_path = '/root/Special_Problem/Special_Problem/Data'
    tile_size = 512
    min_pixel_size = 8
    
    label_map = {'Cluster' : 0, 'Clusters': 0, 'Thyrocyte': 1, 'Thyrocytes': 1}
    
    train_list = (pd.read_csv('/root/Special_Problem/Special_Problem/train_df_summary.csv'))['File'].to_list()
    val_list = (pd.read_csv('/root/Special_Problem/Special_Problem/val_df_summary.csv'))['File'].to_list()
    test_list = (pd.read_csv('/root/Special_Problem/Special_Problem/test_df_summary.csv'))['File'].to_list()    
    
    tile_train_image_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/images/train/"
    tile_train_label_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/labels/train/"
    
    actual_train_image_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/images/train/"
    actual_train_label_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/labels/train/"
   
    tile_valid_image_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/images/valid/"
    tile_valid_label_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/labels/valid/"
    
    actual_valid_image_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/images/valid/"
    actual_valid_label_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/labels/valid/"
    
    tile_test_image_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/images/valid/"
    tile_test_label_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/labels/valid/"
    
    actual_test_image_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/images/test/"
    actual_test_label_path = "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/labels/test/"
    
    directories = ["/root/Special_Problem/Special_Problem/yolo_dataset_version_1/images/train/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/images/val/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/images/test/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/labels/train/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/labels/val/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_1/labels/test/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/images/train/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/images/val/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/images/test/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/labels/train/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/labels/val/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/labels/test/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/tiles/",
    "/root/Special_Problem/Special_Problem/yolo_dataset_version_2/augmented_tiles/"]
    
    transform = A.Compose(
        [
            # Geometric Transformations
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ], p=1),
            # Photometric Transformations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussNoise(std_range=(0.03, 0.05), p=1.0), # 1% to 5% noise
            # Occlusion/regularization
            A.CoarseDropout(
            num_holes_range=(5, 5),
            hole_height_range=(20, 20),
            hole_width_range=(20, 20),
            fill="random_uniform",
            p=0.5),
            A.GridDropout(ratio=0.05, p=0.5)
            # ToTensorV2()
        ],
        seed=42,
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'],)
    )
    
    @classmethod
    def is_supported(cls, ext):
        return ext.lower() in cls.formats
    
    @classmethod
    def get_transform(cls):
        return cls.transform
    
    @staticmethod
    def cluster_group(x):
        if x <= 4:
            return "low"
        elif x <= 10:
            return "medium"
        else:
            return "high"