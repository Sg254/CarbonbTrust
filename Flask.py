
# app_flask.py
from flask import Flask, render_template_string, request, redirect, url_for
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_cnn.h5")

HTML = """
<!doctype html>
<title>MNIST Classifier</title>
<h1>Upload 28x28 image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
  <h2>Prediction: {{pred}} (confidence {{conf:.2f}})</h2>
  <img src="data:image/png;base64,{{img_b64}}">
{% endif %}
"""

def prepare_image(file_stream):
    img = Image.open(file_stream).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr, img

@app.route("/", methods=["GET","POST"])
def index():
    pred = None
    conf = None
    img_b64 = None
    if request.method == "POST":
        f = request.files["file"]
        arr, img = prepare_image(f.stream)
        probs = model.predict(arr)[0]
        p = int(np.argmax(probs))
        pred, conf = p, float(probs[p])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return render_template_string(HTML, pred=pred, conf=conf, img_b64=img_b64)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
