import yt_dlp
from pathlib import Path

def download_youtube_videos(url_list: list, output_dir='input_data'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    downloaded_files = []

    ydl_opts = {'quiet': True,
                'no_warnings': True,
                'outtmpl': str(output_path / '%(title)s.%(ext)s'),
                'format': 'mp4'}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in url_list:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            downloaded_files.append(file_path)

    return downloaded_files


if __name__ == "__main__":
    sample_videos = ['https://www.youtube.com/shorts/h1kUP7xbRcg',
                     'https://www.youtube.com/watch?v=STiDnsmSolE',
                     'https://www.youtube.com/watch?v=HhV-6ONnkig',
                     'https://www.youtube.com/watch?v=1FAj9fJQRZA',
                     'https://www.youtube.com/watch?v=3QSQphrITLo',
                     'https://www.youtube.com/watch?v=Yapi8DjUnjM',
                     'https://www.youtube.com/watch?v=swh448fLd1g']

    result = download_youtube_videos(sample_videos)
    print('Загруженные файлы:')
    for path in result:
        print(path)