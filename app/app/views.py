from app import app
from flask import render_template, jsonify,send_file
import base64
import io
from .ml_model.inference import MyModel



model = MyModel()


@app.route("/")
def index():

    return render_template("home/index.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    img_byte_arr = io.BytesIO()
    model.run().save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img

    # return send_file(model.run(), mimetype='image/gif')
    # results = {"prediction" :"Empty", "probability" :{}}
    #
    # # get data
    # input_img = BytesIO(base64.urlsafe_b64decode(request.form['img']))
    #
    # # model.predict method takes the raw data and output a vector of probabilities
    # res =  model.predict(input_img)
    #
    # results["prediction"] = str(CLASS_MAPPING[np.argmax(res)])
    # results["probability"] = float(np.max(res))*100
    # # results["prediction"] = 5
    # # results["probability"] = 50.424
    #
    # # output data
    # return json.dumps(results)