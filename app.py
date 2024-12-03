from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    model = load_model('models/Banana-mobile.keras')
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predict = model.predict([image])
    label = ['fresh', 'freshUnripe', 'overripe', 'ripe', 'rotten']
    predicted_class = np.argmax(predict, axis=1)[0]

    print()

    return render_template("index.html", prediction=label[predicted_class])

if __name__ == "__main__":
    app.run(debug=True)