import os
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["RESULT_FOLDER"] = "static/results/"

model = YOLO("models/6000box.pt")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            results = model(filepath)
            result_image = results[0].plot()

            result_filepath = os.path.join(app.config["RESULT_FOLDER"], file.filename)
            cv2.imwrite(result_filepath, result_image)

            return redirect(url_for("result", filename=file.filename))
    return render_template("index.html")


@app.route("/result/<filename>")
def result(filename):
    return render_template("result.html", filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
