from transformers import AutoImageProcessor, SiglipForImageClassification
from sentence_transformers import SentenceTransformer

print("Downloading SigLIP...")
AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
SiglipForImageClassification.from_pretrained("Ateeqq/ai-vs-human-image-detector")

print("Downloading CLIP...")
SentenceTransformer("clip-ViT-B-32")

print("All models downloaded.")
