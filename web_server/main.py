"""
Web server script that exposes endpoints and pushes images to Redis for classification by model server. Polls
Redis for response from model server.

Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import io
import json
import os
import time
import uuid

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image
import redis

from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))


def prepare_image(image, target):
    # need image in RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # preprocess image
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.get("/")
def index():
    return "Hello world!"


@app.post("/predict")
def predict(request: Request, img_file: bytes=File(...)):
    data = {"success": False}

    if request.method == "POST":
        image = Image.open(io.BytesIO(img_file))
        image = prepare_image(image,
                              (int(os.environ.get("IMAGE_WIDTH")),
                               int(os.environ.get("IMAGE_HEIGHT")))
                              )

        # ensure NumPy array is C-contiguous, to serialize it later
        image = image.copy(order="C")

        # add the UUID and image to the q
        k = str(uuid.uuid4())
        image = base64.b64encode(image).decode("utf-8")
        d = {"id": k, "image": image}
        db.rpush(os.environ.get("IMAGE_QUEUE"), json.dumps(d))

        # keep looping for CLIENT_MAX_TRIES times
        num_tries = 0
        while num_tries < CLIENT_MAX_TRIES:
            num_tries += 1

            output = db.get(k)

            # check to see if our model has classified the input image
            if output is not None:
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)

                # delete the result from the db and break from the polling loop
                db.delete(k)
                break

            # sleep to give the model a chance to classify the input image
            time.sleep(float(os.environ.get("CLIENT_SLEEP")))

            data["success"] = True
        else:
            raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    return data
