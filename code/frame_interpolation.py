import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from ifrnet.models.IFRNet_S import Model

def read_image(path):
    """Чтение изображения и конвертация в RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def run_ifrnet(input_folder: str,
               output_folder: str,
               model_path: str = 'ifrnet/checkpoints/IFRNET_small/IFRNet_S_Vimeo90K.pth'):
    """
    Запускает IFRNet для интерполяции кадров между всеми последовательными парами.

    :param input_folder: папка с входными кадрами
    :param output_folder: папка для интерполированных кадров
    :param model_path: путь к претренированной модели IFRNet
    """

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"IFRNet model not found: {model_path}")

    os.makedirs(output_folder, exist_ok=True)

    print("\nRunning IFRNet frame interpolation...")

    # Используем CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Загружаем модель
    model = Model().to(device).eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Проверяем структуру чекпоинта
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"Model loaded from: {model_path}")

    # Получаем список всех изображений и сортируем по имени
    image_paths = sorted(glob.glob(os.path.join(input_folder, '*')))
    
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images for interpolation, found {len(image_paths)}")
    
    print(f"Found {len(image_paths)} images in {input_folder}")
    
    # Обрабатываем каждую пару последовательных кадров
    for i in range(len(image_paths) - 1):
        img0_path = image_paths[i]
        img1_path = image_paths[i + 1]
        
        print(f"Processing pair {i+1}/{len(image_paths)-1}: {osp.basename(img0_path)} -> {osp.basename(img1_path)}")
        
        try:
            # Читаем изображения
            img0_np = read_image(img0_path)
            img1_np = read_image(img1_path)
            
            # Преобразуем в тензоры и нормализуем
            img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
            img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
            
            # Создаем временной эмбеддинг для середины между кадрами
            embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(device)
            
            # Выполняем интерполяцию
            with torch.no_grad():
                imgt_pred = model.inference(img0, img1, embt)
            
            # Конвертируем обратно в numpy
            imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            
            # Конвертируем из RGB обратно в BGR для OpenCV
            imgt_pred_bgr = cv2.cvtColor(imgt_pred_np, cv2.COLOR_RGB2BGR)
            img0_bgr = cv2.cvtColor(img0_np, cv2.COLOR_RGB2BGR)
            
            # Сохраняем оба кадра в правильном порядке
            # 1. Сохраняем оригинальный кадр с правильным именем
            frame_num_0 = i * 2  # Четные номера для оригинальных кадров
            orig_output_name = f"frame_{frame_num_0:06d}.png"
            orig_output_path = osp.join(output_folder, orig_output_name)
            cv2.imwrite(orig_output_path, img0_bgr)
            
            # 2. Сохраняем интерполированный кадр
            frame_num_interp = i * 2 + 1  # Нечетные номера для интерполированных
            interp_output_name = f"frame_{frame_num_interp:06d}.png"
            interp_output_path = osp.join(output_folder, interp_output_name)
            cv2.imwrite(interp_output_path, imgt_pred_bgr)
            
            print(f"  Saved: {orig_output_name} and {interp_output_name}")
            
        except Exception as e:
            print(f"  Error processing pair {img0_path} -> {img1_path}: {e}")
            continue
    
    # Сохраняем последний оригинальный кадр
    if len(image_paths) > 0:
        last_img_path = image_paths[-1]
        last_img_np = read_image(last_img_path)
        last_output_name = f"frame_{(len(image_paths) - 1) * 2:06d}.png"
        last_output_path = osp.join(output_folder, last_output_name)
        cv2.imwrite(last_output_path, cv2.cvtColor(last_img_np, cv2.COLOR_RGB2BGR))
        print(f"Saved last frame: {last_output_name}")
    
    total_frames = len(glob.glob(osp.join(output_folder, '*.png')))
    print(f"Total frames saved: {total_frames}")
    print("IFRNet finished successfully.")
    
    return total_frames


if __name__ == '__main__':
    import sys
    from video_handler import video_to_photos, photos_to_video, cleanup_folders
    
    INPUT_FOLDER = './input_frames'
    INTERPOLATED_FRAMES = './frames_2_interpolated'
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_video> <output_video>")
        sys.exit(1)
    
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    
    try:
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        os.makedirs(INTERPOLATED_FRAMES, exist_ok=True)
        
        # 1. Извлекаем кадры из видео и получаем FPS
        print("Extracting frames from video...")
        fps = video_to_photos(input_video_path, INPUT_FOLDER)
        
        # 2. Интерполяция кадров (IFRNet)
        total_frames = run_ifrnet(input_folder=INPUT_FOLDER, 
                                 output_folder=INTERPOLATED_FRAMES)
        
        print("\nEncoding final video...")
        
        # 3. Создаем видео из интерполированных кадров
        # Оставляем тот же FPS, но количество кадров удвоилось
        photos_to_video(output_video_path, fps, INTERPOLATED_FRAMES)
        
        # Рассчитываем длительность
        original_duration = total_frames / (fps * 2)  # total_frames включает интерполированные
        output_duration = total_frames / fps
        
        print(f"\nProcess completed successfully!")
        print(f"Output video: {output_video_path}")
        print(f"Original FPS: {fps}")
        print(f"Original frame count: ~{total_frames // 2}")
        print(f"Output frame count: {total_frames}")
        print(f"Duration preserved: {output_duration:.2f}s (same as input)")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Очистка временных файлов
        print("\nCleaning up temporary frames...")
        cleanup_folders()