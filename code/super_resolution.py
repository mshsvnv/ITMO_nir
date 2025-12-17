import os
import subprocess

def run_real_esrgan(input_folder: str,
                    output_folder: str,
                    exe_path: str = 'realesrgan/realesrgan-ncnn-vulkan.exe',
                    scale: int = 2):
    """
    Запускает Real-ESRGAN (ncnn-vulkan) для апскейла кадров.

    :param input_folder: папка с входными кадрами
    :param output_folder: папка для SR-кадров
    :param exe_path: путь к realesrgan-ncnn-vulkan.exe
    :param scale: коэффициент апскейла (2, 3, 4)
    """

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Real-ESRGAN executable not found: {exe_path}")

    os.makedirs(output_folder, exist_ok=True)

    cmd = [exe_path, 
           '-i', input_folder,
           '-o', output_folder,
           '-s', str(scale)]

    print("\nRunning Real-ESRGAN Super Resolution...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError("Real-ESRGAN failed with non-zero exit code")

    print("Real-ESRGAN finished successfully.")