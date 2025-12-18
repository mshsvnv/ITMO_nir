import os
import sys
import subprocess

from super_resolution import run_real_esrgan
from video_handler import video_to_photos, photos_to_video, cleanup_folders

INPUT_FOLDER = './input_frames'
INTERPOLATED_FRAMES = './frames_2_interpolated'
UPSCALED_FRAMES = './frames_3_upscaled'


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_video> <output_video>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]

    try:
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        os.makedirs(INTERPOLATED_FRAMES, exist_ok=True)
        os.makedirs(UPSCALED_FRAMES, exist_ok=True)

        # 1. Извлекаем кадры из видео и получаем FPS
        fps = video_to_photos(input_video_path)

        # 2. Запускаем интерполяцию кадров
        print("\nRunning frame interpolation...")
        
        # сама интерполяция
        subprocess.run(['./ifrnet-cpu/ifrnet-ncnn-vulkan.exe', 
                        '-i', INPUT_FOLDER, 
                        '-o', INTERPOLATED_FRAMES],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        
        # Super Resolution (Real-ESRGAN)
        print("\nRunning Super Resolution (Real-ESRGAN)...")
        
        run_real_esrgan(input_folder=INTERPOLATED_FRAMES, 
                        output_folder=UPSCALED_FRAMES, 
                        scale=2)
        
        print("\nEncoding final video...")

        photos_to_video(output_video_path, fps, UPSCALED_FRAMES)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")

    finally:
        # Очистка временных файлов
        print("Cleaning up temporary frames...")
        cleanup_folders() 