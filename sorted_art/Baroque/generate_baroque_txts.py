#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    folder = Path(__file__).parent
    metadata_path = folder / 'metadata.json'
    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)
    # map hash to caption
    caption_map = {item['hash']: item['caption'] for item in metadata}

    for jpg in folder.glob('*.jpg'):
        base = jpg.stem
        caption = caption_map.get(base)
        if caption is None:
            print(f"Warning: no caption for {jpg.name}")
            continue
        txt_path = folder / f"{base}.txt"
        with txt_path.open('w', encoding='utf-8') as out:
            out.write(caption)
        print(f"Created {txt_path.name}")

if __name__ == '__main__':
    main()
