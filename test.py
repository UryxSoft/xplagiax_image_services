import os
os.environ["HF_HOME"] = "/tmp/hf"

from app.models.registry import ModelRegistry

m = ModelRegistry("Ateeqq/ai-vs-human-image-detector", "sentence-transformers/clip-ViT-B-32", "cpu")
try:
    m.load_all()
    print("Loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
