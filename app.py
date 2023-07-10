from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt

app = Flask(__name__)

# Página de inicio del formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar el formulario
@app.route('/process', methods=['POST'])
def process():
    # Obtener las imágenes enviadas por el formulario
    content_image = request.files['content_image']
    style_image = request.files['style_image']
    
    # Abrir las imágenes utilizando PIL
    content_image = Image.open(content_image)
    style_image = Image.open(style_image)
    
    # Procesar las imágenes
    content_image_arr = np.array(content_image) / 255.0
    style_image_arr = np.array(style_image) / 255.0
    
    content_image_arr = content_image_arr.astype(np.float32)[np.newaxis, ...]
    style_image_arr = style_image_arr.astype(np.float32)[np.newaxis, ...]

    style_image_arr = tf.image.resize(style_image_arr, (256, 256))

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(tf.convert_to_tensor(content_image_arr), tf.convert_to_tensor(style_image_arr))
    stylized_image = outputs[0]

    # Mostrar la imagen generada
    plt.imshow(stylized_image)
    plt.title('Generated Image')
    plt.axis('off')
    plt.show()
    
    return 'Procesamiento completado'

if __name__ == '__main__':
    app.run()
