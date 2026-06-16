import os
import random
import kagglehub
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class DF2KDataset(Dataset):
    def __init__(self, split='train', scale=2, downgrade='unknown', lr_patch_size=32):
        """
        lr_patch_size: Размер вырезаемого куска для LR картинки. 
                       Для HR картинки размер будет lr_patch_size * scale.
        """
        assert split in ['train', 'val']
        assert downgrade in ['bicubic', 'unknown']
        
        self.split = split
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        
        base_path = kagglehub.dataset_download("anvu1204/df2kdata", output_dir="./data/df2k")
        folder_suffix = 'train' if split == 'train' else 'valid'
        
        self.hr_dir = os.path.join(base_path, f'DF2K_{folder_suffix}_HR')
        
        if downgrade == 'bicubic':
            lr_base = os.path.join(base_path, f'DF2K_{folder_suffix}_LR_bicubic')
        else:
            lr_base = os.path.join(base_path, f'DF2K_{folder_suffix}_LR_unknown')
            
        possible_lr_dir = os.path.join(lr_base, f'X{scale}')
        if os.path.exists(possible_lr_dir):
            self.lr_dir = possible_lr_dir
        else:
            self.lr_dir = lr_base

        self.hr_filenames = sorted([
            f for f in os.listdir(self.hr_dir) 
            if f.lower().endswith(('.png', '.jpg'))
        ])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, idx):
        hr_name = self.hr_filenames[idx]
        hr_path = os.path.join(self.hr_dir, hr_name)
        
        name_without_ext, ext = os.path.splitext(hr_name)
        lr_name = f"{name_without_ext}x{self.scale}{ext}"
        lr_path = os.path.join(self.lr_dir, lr_name)
        
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')
        
        # Только для тренировки вырезаем случайные патчи
        if self.split == 'train':
            lr_w, lr_h = lr_image.size
            
            # Генерируем случайные координаты для вырезания LR
            # Убедимся, что картинка не меньше патча
            max_x = max(0, lr_w - self.lr_patch_size)
            max_y = max(0, lr_h - self.lr_patch_size)
            
            lr_x = random.randint(0, max_x)
            lr_y = random.randint(0, max_y)
            
            # Вычисляем соответствующие координаты для HR
            hr_x = lr_x * self.scale
            hr_y = lr_y * self.scale
            hr_patch_size = self.lr_patch_size * self.scale
            
            # Кропаем изображения
            lr_image = lr_image.crop((lr_x, lr_y, lr_x + self.lr_patch_size, lr_y + self.lr_patch_size))
            hr_image = hr_image.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))
            
            # Аугментация: Случайное отзеркаливание
            if random.random() > 0.5:
                lr_image = TF.hflip(lr_image)
                hr_image = TF.hflip(hr_image)
            if random.random() > 0.5:
                lr_image = TF.vflip(lr_image)
                hr_image = TF.vflip(hr_image)

        hr_tensor = self.to_tensor(hr_image)
        lr_tensor = self.to_tensor(lr_image)
        
        return lr_tensor, hr_tensor


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    scale_factor = 2
    test_dataset = DF2KDataset(split='train', scale=scale_factor, downgrade='unknown')
    
    print(f"Всего изображений в сплите: {len(test_dataset)}")
    print(f"Путь HR: {test_dataset.hr_dir}")
    print(f"Путь LR: {test_dataset.lr_dir}")
    print("-"*50)

    # соответствие имен файлов
    print("Проверка маппинга имен файлов:")
    for i in range(min(5, len(test_dataset))):
        hr_name = test_dataset.hr_filenames[i]
        name_without_ext, ext = os.path.splitext(hr_name)
        expected_lr_name = f"{name_without_ext}x{scale_factor}{ext}"

        hr_exists = os.path.exists(os.path.join(test_dataset.hr_dir, hr_name))
        lr_exists = os.path.exists(os.path.join(test_dataset.lr_dir, expected_lr_name))
        
        print(f"  Пара {i+1}:")
        print(f"    -> HR: {hr_name} (Существует: {hr_exists})")
        print(f"    -> LR: {expected_lr_name} (Существует: {lr_exists})")

    print("-"*50)

    # тензоры
    print("Анализ тензоров первого элемента:")
    lr_tensor, hr_tensor = test_dataset[0]
    
    print(f"  Размер LR тензора [C, H, W]: {list(lr_tensor.shape)}")
    print(f"  Размер HR тензора [C, H, W]: {list(hr_tensor.shape)}")
    print(f"  Тип данных: {lr_tensor.dtype}")
    print(f"  Диапазон значений LR: [{lr_tensor.min().item():.2f}, {lr_tensor.max().item():.2f}]")
    
    # проверка масштаба
    calculated_scale_h = hr_tensor.shape[1] / lr_tensor.shape[1]
    calculated_scale_w = hr_tensor.shape[2] / lr_tensor.shape[2]
    print(f"  Фактический масштаб по высоте: {calculated_scale_h}x (Ожидалось: {scale_factor}x)")
    print(f"  Фактический масштаб по ширине: {calculated_scale_w}x (Ожидалось: {scale_factor}x)")