import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from PIL import Image

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from piq import DISTS

from RRDBNet_arch import RRDBNet
from dataloader import DF2KDataset

def validate():
    parser = argparse.ArgumentParser(description='ESRGAN Validation/Testing Script')
    parser.add_argument('--weights', type=str, default='./checkpoints/best_esrgan.pth', 
                        help='Путь к весам модели')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Масштаб увеличения (2, 3 или 4)')
    parser.add_argument('--save_num', type=int, default=10, 
                        help='Количество визуальных результатов для сохранения')
    parser.add_argument('--output_dir', type=str, default='./validation_results', 
                        help='Папка для сохранения визуальных результатов')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"устройство: {device}")

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23).to(device)
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Веса не найдены по пути: {args.weights}")
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"[+] Успешно загружены веса: {args.weights}")

    val_dataset = DF2KDataset(split='val', scale=args.scale)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Найдено {len(val_loader)} изображений в валидационном сете.")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    dists_metric = DISTS().to(device)

    total_psnr, total_ssim, total_lpips, total_dists = 0.0, 0.0, 0.0, 0.0
    
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Цикл валидации
    with torch.no_grad():
        for idx, (lr_imgs, hr_imgs) in enumerate(tqdm(val_loader, desc="Validation")):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            sr_imgs = torch.clamp(model(lr_imgs), 0, 1)

            total_psnr += psnr_metric(sr_imgs, hr_imgs).item()
            total_ssim += ssim_metric(sr_imgs, hr_imgs).item()
            total_dists += dists_metric(sr_imgs, hr_imgs).item()

            total_lpips += lpips_metric(sr_imgs * 2 - 1, hr_imgs * 2 - 1).item()

            if idx < args.save_num:
                lr_resized = torch.nn.functional.interpolate(lr_imgs, size=hr_imgs.shape[2:], mode='nearest')
                
                # апскейл соседа | Результат ESRGAN | Оригинал (HR)
                grid = make_grid(torch.cat([lr_resized, sr_imgs, hr_imgs], dim=0), nrow=3, padding=4)
                save_path = os.path.join(args.output_dir, f"result_sample_{idx:03d}.png")
                save_image(grid, save_path)

    num_samples = len(val_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    avg_dists = total_dists / num_samples

    print("\n")
    print(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (Scale: X{args.scale})")
    print(f"Средний PSNR:  {avg_psnr:.2f} dB  (Выше — лучше)")
    print(f"Средний SSIM:  {avg_ssim:.4f}     (Ближе к 1 — лучше)")
    print(f"Средний LPIPS: {avg_lpips:.4f}    (Ниже — лучше, перцептивная)")
    print(f"Средний DISTS: {avg_dists:.4f}    (Ниже — лучше, текстурная)")


if __name__ == '__main__':
    validate()