def is_rtsp(media_link : str) -> bool:
    if media_link.startswith('rtsp://'):
        return True
    return False


def is_video_file(media_link: str) -> bool:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(media_link.lower().endswith(ext) for ext in video_extensions)

