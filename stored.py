import json
from core import Embedding

embed = Embedding(collection_name='cve-detection')

with open('output/tracking.json', 'r') as f:
    tracks = json.load(f)

for track_id, payload in tracks.items():
    img_path = payload.get('filename')
    if img_path:
        embed.store_embedding((int(track_id)), img_path, payload)