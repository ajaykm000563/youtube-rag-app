# youtubedata.py

import yt_dlp
import requests
import json

def Get_Youtube_Data(url: str) -> str:
    """
    Uses your exact code, just receives full YouTube URL from app.py
    """

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

        subtitles = info.get("subtitles", {}).get("en")
        if not subtitles:
            subtitles = info.get("automatic_captions", {}).get("en")

        if subtitles:
            subtitle_url = subtitles[0]["url"]
            raw = requests.get(subtitle_url).text
            print("Transcript successfully extracted.\n")
        else:
            raise RuntimeError("No English subtitles found.")

    data = json.loads(raw)

    text_parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            if "utf8" in seg:
                text_parts.append(seg["utf8"])

    text = " ".join(text_parts)
    return text
