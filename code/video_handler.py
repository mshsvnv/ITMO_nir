import cv2
import os
import glob
import shutil

INPUT_FOLDER = './input_frames'
OUTPUT_FOLDER = './output_frames'


def video_to_photos(video_path, output_folder=INPUT_FOLDER):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: can't open file!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    frame_count = 0
    saved_count = 0
    
    print("\nStarted downloading video...")
    while True:
        ret, frame = cap.read()
        
        if ret:
            frame_filename = os.path.join(output_folder, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

            if (frame_count + 1) % 100 == 0 or (frame_count + 1) == total_frames:
                percent = (frame_count + 1) / total_frames * 100
                print(f'Processed {frame_count + 1}/{total_frames} frames ({percent:.1f}%)')
            
            frame_count += 1
        else:
            break
    
    cap.release()
    print(f"Successfully downloaded to '{output_folder}'!\n")

    return fps


def photos_to_video(video_path, original_fps, 
                    interpolated_frames_folder=OUTPUT_FOLDER):
    
    all_frames = sorted(glob.glob(os.path.join(interpolated_frames_folder, "*.png")))
    total_frames = len(all_frames)
    
    new_fps = original_fps * 2
    first_frame = cv2.imread(all_frames[0])
    if first_frame is None:
        
        first_frame_rgb = cv2.imread(all_frames[0], cv2.IMREAD_UNCHANGED)
        if first_frame_rgb is not None:
            first_frame = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            print("Error: can't read first frame!")
            return None
    
    height, width = first_frame.shape[:2]
    
    print(f"  Resolution: {width}x{height}")
    print(f"  New FPS: {new_fps} (old: {original_fps})")
    print(f"  Total frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, new_fps, (width, height))
     
    print("\nStarted recording video...")
    
    for i, frame_path in enumerate(all_frames):
        frame = cv2.imread(frame_path)
        
        if frame is None:
            frame_rgb = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if frame_rgb is not None:
                if len(frame_rgb.shape) == 2:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
                else:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if frame is not None:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
        
        if (i + 1) % 100 == 0 or (i + 1) == total_frames:
            percent = (i + 1) / total_frames * 100
            print(f'Processed {i + 1}/{total_frames} frames ({percent:.1f}%)')
    
    out.release()
    print(f"Video created: '{video_path}'")


def cleanup_folders():
    """
    Удаляет папки input_frames и output_frames после завершения обработки.
    """
    folders_to_clean = [INPUT_FOLDER, OUTPUT_FOLDER]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                # Удаляем всю папку со всем содержимым
                shutil.rmtree(folder)
                print(f"Successfully removed folder: '{folder}'")
            except Exception as e:
                print(f"Error removing folder '{folder}': {e}")