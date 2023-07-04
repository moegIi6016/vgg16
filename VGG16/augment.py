import os
import cv2
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast, GaussianBlur

def augment(image_path, save_dir):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmentations = [
        # ('horizontal_flip', HorizontalFlip(p=1)),
        # ('vertical_flip', VerticalFlip(p=1)),
        # ('rotate', Rotate(limit=25, p=1)),
        ('random_brightness_contrast_1', RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)),
        ('random_brightness_contrast_2', RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1)),
        ('random_brightness_contrast_3', RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)),
        ('gaussian_blur', GaussianBlur(p=1, blur_limit=(3, 7))),
    ]

    class_name = os.path.basename(os.path.dirname(image_path))

    for name, augmentation in augmentations:
        augmented = augmentation(image=image)
        augmented_image = augmented['image']

        #Resize
        resized_image = cv2.resize(augmented_image, (224, 224))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

        #保存
        save_path = os.path.join(save_dir, class_name, f'{name}_{os.path.basename(image_path)}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, resized_image)

        resized_original = cv2.resize(image, (224, 224))
        resized_original = cv2.cvtColor(resized_original, cv2.COLOR_RGB2BGR)
        original_save_path = os.path.join(save_dir, class_name, os.path.basename(image_path))
        cv2.imwrite(original_save_path, resized_original)

input_dir = 'C:/Users/user/Desktop/6.30vgg/dataset'
output_dir = 'C:/Users/user/Desktop/VGG16_1/augmented'

image_files = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        lowercase_filename = file.lower()
        if lowercase_filename.endswith('.jpg') or lowercase_filename.endswith('.jpeg') or lowercase_filename.endswith(
                '.png'):
            image_files.append(os.path.join(root, file))
for image_file in image_files:
    augment(image_file, output_dir)