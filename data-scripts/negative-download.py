import time
import os
from icrawler.builtin import BingImageCrawler

actors = [
    #"Ryan Reynolds",
    #"Jake Gyllenhaal",
    #"Bradley Cooper",
    #"Chris Evans",
    #"Leonardo DiCaprio"
    #"a mans face",
    #"a black mans face",
    #"an asian mans face"
    "a womans face"
]



modifiers = [
    "",
    #"face",
    #"side",
    #"portrait",
    #"movie still",
    #"young",
    #"interview",
    #"close up",
    #"red carpet",
    #"Oscars",
    #"premiere"
]

BASE_OUTPUT_DIR = "../raw_images/negative"
IMAGES_PER_QUERY = 300
SLEEP_SECONDS = 1



def count_existing_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])




os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

for actor in actors:
    for mod in modifiers:
        query = f"{actor} {mod}".strip()

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

