import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path

import tensorflow as tf

flags = tf.compat.v1.flags
flags.DEFINE_string('input_path', '', 'commons/uploads/2022-07-01T16.57.46.848449_input_file.filename.jpg')
FLAGS = flags.FLAGS

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./commons/features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./commons/image") / (feature_path.stem + ".jpg"))
features = np.array(features)

if True:

        # Save query image
        img = Image.open(FLAGS.input_path)  # PIL image
        uploaded_img_path = "commons/uploads/" + datetime.now().isoformat().replace(":", ".") + "_" + "input_file.filename.jpg"
        img.save(uploaded_img_path)
        

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        for score in scores:
            if(score[0]<=0.65):
                print("Match found: difference= "+str(score[0])+" Match =: database_cattle"+(str(score[1]).split(sep="database_cattle")[1]).split(".")[0])
                break
        else :print("no match found")  


