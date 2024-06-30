# 基于Flask的低俗图片检测Web应用程序

本文档提供了一个使用Flask处理图像上传、预处理图像，并使用TensorFlow模型进行预测的Web应用程序的概述和说明。

## 概述

该Web应用程序允许用户上传图像，然后对图像进行处理并使用预训练的TensorFlow模型进行评估。该应用程序使用Flask，一个轻量级的Python
Web框架，来处理HTTP请求和响应。

## 功能

- 接受`png`、`jpg`和`jpeg`格式的图像上传。
- 对图像进行预处理以进行评估。
- 加载并使用TensorFlow模型进行预测。
- 以JSON格式提供结果。

## 要求

- Flask
- TensorFlow
- Werkzeug

## 设置

1. **安装依赖**：确保在你的环境中安装了Flask、TensorFlow和Werkzeug。
   ```sh
   pip install Flask tensorflow werkzeug
   ```
2. **目录结构**：
   ```
   project/
   ├── app.py
   ├── uploads/
   ├── model/
   │   └── saved_model
   └── templates/
       └── index.html
   ```
    - `app.py`：主应用程序文件。
    - `uploads/`：存储上传图像的目录。
    - `model/saved_model`：包含保存的TensorFlow模型的目录。
    - `templates/index.html`：前端HTML文件。

3. **运行应用程序**：
   ```sh
   python app.py
   ```
   运行app.py即可，然后打开127.0.0.1:5000进入前端。
   一次支持上传单张/多张图片进行推理，推理出程度>=60%会被标红，<=20%会标绿，在之间的会标黄
   ![img1.png](screenshots%2Fimg1.png)

## 代码解释

### 配置

```python
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
```

- `UPLOAD_FOLDER`：存储上传图像的目录。
- `ALLOWED_EXTENSIONS`：允许上传的文件扩展名集合。

### 辅助函数

#### `allowed_file(filename)`

检查文件是否具有允许的扩展名。

```python
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
```

#### `read_image(filename)`

加载并预处理图像以进行TensorFlow模型的评估。

```python
def read_image(filename: str) -> tf.Tensor:
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    image = preprocess_for_evaluation(image, 480, tf.float16)
    image = tf.reshape(image, -1)
    return image
```

### 路由

#### 首页路由

渲染主HTML页面。

```python
@app.route('/')
def index():
    return render_template('/index.html')
```

#### 预测路由

处理文件上传，预处理图像，运行TensorFlow模型并返回预测结果。

```python
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

            # 读取并预处理图像
            image = read_image(filepath)

            # 确保使用GPU进行推理
            with tf.device('/GPU:0'):
                preds = model([image])
                probability = 100 * tf.get_static_value(preds[0])[0]

            results.append({'probability': probability, 'filename': filename})
        else:
            results.append({'error': '文件类型不允许'})

    return jsonify(results)
```

## 注意事项

- **模型加载**：TensorFlow模型从`./model/saved_model`加载。确保模型放置正确。
- **GPU利用**：应用程序尝试使用GPU进行推理。如果GPU不可用，请根据需要进行调整。
- **错误处理**：对文件上传和预测实现了基本的错误处理。
