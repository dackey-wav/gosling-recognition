import time
import os
from icrawler.builtin import BingImageCrawler


ACTOR = "Ryan Gosling"

modifiers = [
    "",
    "Grey man",
    "drive",
    "barbie",
    "blade runner",
    "place under the pines",
    "the notebook,"
    "face",
    "side view",
    "portrait",
    "movie still",
    "young",
    "old",
    "interview",
    "close up",
    "red carpet",
    "Oscars",
    "Cannes",
    "premiere",
    "smiling",
    "serious",
    "beard",
    "glasses"
]

BASE_OUTPUT_DIR = "../raw_images/gosling"
IMAGES_PER_QUERY = 100
SLEEP_SECONDS = 1


def count_existing_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

for mod in modifiers:
    query = f"{ACTOR} {mod}".strip()

    existing_count = count_existing_images(BASE_OUTPUT_DIR)

    print(f"\nDownloading: '{query}'")
    print(f"Existing images: {existing_count}")
    print(f"File index offset: {existing_count}")

    crawler = BingImageCrawler(
        storage={"root_dir": BASE_OUTPUT_DIR}
    )

    crawler.crawl(
        keyword=query,
        max_num=IMAGES_PER_QUERY,
        file_idx_offset=existing_count
    )

    print(f"Sleeping {SLEEP_SECONDS} seconds...\n")
    time.sleep(SLEEP_SECONDS)
