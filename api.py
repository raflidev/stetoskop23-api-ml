from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import script as model
app = Flask(__name__)

@app.route("/")
def home():
    return "<p>Hello, World!</p>"

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):    
    return send_file(filename, as_attachment=True)

@app.route('/predict', methods = ['GET', 'POST'])
def cek():
  if request.method == 'POST':
    f = request.files['file']
    f.save(secure_filename(f.filename))
    return model.predict(f.filename)

if __name__ == "__name__":
  app.run(debug=True)

# flask --app api run