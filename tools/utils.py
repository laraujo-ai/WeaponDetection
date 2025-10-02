def is_rtsp(media_link: str) -> bool:
    """Check if the media link is an RTSP stream.

    Args:
        media_link: URL or path to media source.

    Returns:
        bool: True if the link is an RTSP stream, False otherwise.
    """
    return media_link.startswith('rtsp://')


def is_video_file(media_link: str) -> bool:
    """Check if the media link is a video file based on extension.

    Args:
        media_link: Path or URL to media source.

    Returns:
        bool: True if the link ends with a recognized video extension, False otherwise.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    return media_link.lower().endswith(video_extensions)


def _process_class_names(class_names):
    """Process class names from command-line arguments.

    Handles both list format and comma-separated string format.

    Args:
        class_names: List of class names or a single comma-separated string.

    Returns:
        list: Processed list of class names with whitespace stripped.

    """
    if class_names and len(class_names) == 1 and "," in class_names[0]:
        return [c.strip() for c in class_names[0].split(",") if c.strip()]
    return class_names or []
