import albumentations as A
class StaticVariable:
    formats = ['.jpeg', '.jpg', '.png']
    data_path = '/root/Special_Problem/Special_Problem/Data'
    tile_size = 512
    min_pixel_size = 8
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
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0), # 1% to 5% noise
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