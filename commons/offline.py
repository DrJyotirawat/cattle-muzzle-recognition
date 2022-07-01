import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()
    for img_path in sorted(Path("commons/image").glob("*.jpg")):
        print("found: "+str(img_path))  # e.g., ./static/img/xxx.jpg
        img=Image.open(img_path)
        feature = fe.extract(img)
        feature_path = Path("commons/features") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)