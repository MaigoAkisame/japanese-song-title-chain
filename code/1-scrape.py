import json
import os
import requests
import sys
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

"""
Scraper to get song title and artists from uta-net.com.
Songs successfully scraped will be saved to the OUTPUT_FILE.
You can run this script repeatedly until all songs are successfully scraped.
"""

BASE_URL = "https://www.uta-net.com/song/{}"
OUTPUT_FILE = "../data/songs.jsonl"
MAX_SONG_ID = 377965
NUM_WORKERS = 20


def fetch_song_info(song_id: int) -> dict[str, Any]:
    url = BASE_URL.format(song_id)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {"id": song_id, "status": resp.status_code}
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.select_one("h2.kashi-title")
        artist_tag = soup.select_one('span[itemprop="byArtist name"]')
        return {
            "id": song_id,
            "title": title_tag.get_text(strip=True) if title_tag else None,
            "artist": artist_tag.get_text(strip=True) if artist_tag else None,
        }
    except requests.RequestException as e:
        print(f"FAILED TO FETCH SONG {song_id}", file=sys.stderr)
        return None


if __name__ == "__main__":
    completed_songs = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                js = json.loads(line)
                completed_songs.add(js["id"])

    song_ids = sorted(set(range(1, MAX_SONG_ID + 1)) - completed_songs)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(fetch_song_info, sid): sid for sid in song_ids}
        for future in as_completed(futures):
            data = future.result()
            if data is not None:
                print(data)
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
