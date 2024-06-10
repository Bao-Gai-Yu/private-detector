import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from private_detector.utils.preprocess import preprocess_for_evaluation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def read_image(filename: str) -> tf.Tensor:
    """
    Load and preprocess image for inference with the Private Detector
    """
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    image = preprocess_for_evaluation(image, 480, tf.float16)
    image = tf.reshape(image, -1)
    return image


@app.route('/')
def index():
    return render_template('/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})

    files = request.files.getlist('files')
    results = []

    # 加载模型
    model = tf.saved_model.load('./model/saved_model')

    for file in files:
        if file.filename == '':
            results.append({'error': 'No selected file'})
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 读入图片
            image = read_image(filepath)

            # # 得到模型返回结果
            # preds = model([image])
            # probability = 100 * tf.get_static_value(preds[0])[0]
            # 确保使用GPU进行推理
            with tf.device('/GPU:0'):
                preds = model([image])
                probability = 100 * tf.get_static_value(preds[0])[0]

            results.append({'probability': probability, 'filename': filename})
        else:
            results.append({'error': '文件类型不允许'})

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
