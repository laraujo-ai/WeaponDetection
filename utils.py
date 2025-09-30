import argparse

def is_video_file(file_path: str) -> bool:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)

