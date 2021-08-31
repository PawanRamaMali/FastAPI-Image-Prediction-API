"""
Model server script that polls Redis for images to classify

Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import json
import os
import sys
import time

from keras.applications import ResNet50
from keras.applications import imagenet_utils
import numpy as np
import redis

db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

# load the pre-trained Keras model (easy to substitute for your own)
model = ResNet50(weights="imagenet")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using
    # the supplied data type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    return a


def classify_process():
    # continually poll for new images to classify
    while True:
        # pop off multiple images from Redis queue
        with db.pipeline() as pipe:
            pipe.lrange(os.environ.get("IMAGE_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("IMAGE_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queue, _ = pipe.execute()

        image_ids = []
        batch = None
        for q in queue:
            # deserialize the object and get the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"],
                                        os.environ.get("IMAGE_DTYPE"),
                                        (1, int(os.environ.get("IMAGE_HEIGHT")),
                                         int(os.environ.get("IMAGE_WIDTH")),
                                         int(os.environ.get("IMAGE_CHANS")))
                                        )

            # stack the data if we have a batch to process
            batch = image if batch is None else np.vstack([batch, image])

            # update the list of image_ids
            image_ids.append(q["id"])

        # do we need to process the batch?
        if len(image_ids) > 0:
            # classify the batch
            print("* Batch size: {}".format(batch.shape))
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds)

            # format the results and write to db
            for (image_id, results_for_image) in zip(image_ids, results):
                output = []

                for (label_id, label, prob) in results_for_image:
                    r = {"label": label, "probability": float(prob)}
                    output.append(r)

                # store the output predictions in the database,
                # use image_id as the key so we can fetch the results
                db.set(image_id, json.dumps(output))

        time.sleep(float(os.environ.get("SERVER_SLEEP")))

if __name__ == "__main__":
    classify_process()
