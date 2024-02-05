import numpy as np
import pdf2image
from PIL import Image
import layoutparser as lp
import io
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import requests

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for routes starting with /api/

# Define your API endpoint
@app.route('/analyze-pdf', methods=['POST'])
def analyze():

    uploaded_file = request.files['pdf_file']
    pdf_bytes = uploaded_file.read()

    images = pdf2image.convert_from_bytes(pdf_bytes)
    img = np.asarray(images[0])
    
    # img = np.asarray(pdf2image.convert_from_path(uploaded_file.read())[0])

    # Initialize the layout models
    model1 = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5], label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})

    # Detect layout elements
    layout_result1 = model1.detect(img)


    # Draw bounding boxes on each image for the corresponding model
    result_1 = lp.draw_box(img, layout_result1, box_width=5, box_alpha=0.2, show_element_type=True)

    # Convert the PIL image to a byte stream (JPEG format)
    img_byte_array = io.BytesIO()
    result_1.save(img_byte_array, format='JPEG')
    img_byte_array.seek(0)

    print("Finished analyzing file")

    # Return the byte stream as a response
    return send_file(img_byte_array, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)