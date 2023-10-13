from PIL import Image
import requests
import json
import sys
from pathlib import Path

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

arg = sys.argv[1]

if Path(arg).exists():
    image = Image.open(Path(arg))
else:
    url = arg
    image = Image.open(requests.get(url, stream=True).raw)

with open("mapping.json", "r") as mf:
    mapping = json.load(mf)


def probe(choices, image):
    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    (probs,) = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    return sorted(zip(probs.numpy(force=True).tolist(), choices), reverse=True)


print("Guessing country...")
countries = list(mapping.keys())
country_rank = probe(list(f'A street view photo in the country of {c}' for c in countries), image)
print(country_rank[:5])
country = country_rank[0][1][len('A street view photo in the country of '):]
print(f"checking cities in {country}...")
cities_choices = mapping[country]
city_rank = probe(list(f'A street view photo near {c} in the country of {country}' for c in cities_choices), image)
print(city_rank)
print()
print(city_rank[0][1])