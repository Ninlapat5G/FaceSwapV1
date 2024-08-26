import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import tempfile
import base64
import threading
import uuid
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# กำหนด Gmail ต้นทาง
SENDER_EMAIL = "patcharadanai-s@rmutp.ac.th"
SENDER_PASSWORD = "dead254501"  # ใช้ App Password สำหรับความปลอดภัย

class FaceSwapApp:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = get_model('inswapper_128.onnx', download=False, download_zip=False)
        self.face_enhancer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=1, arch='clean')
        self.progress = 0
        self.processing = False
        self.result = None
        self.result_id = None

    def swap_face(self, src_img, dst_img, src_face, dst_face):
        swapped_img = self.swapper.get(dst_img, dst_face, src_face, paste_back=True)
        return swapped_img

    def enhance_image(self, img):
        _, _, enhanced_img = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        return enhanced_img

    def process_images(self, img1_path, img2_path, num_faces):
        self.progress = 0
        self.processing = True
        self.result = None
        self.result_id = None

        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            face1 = self.app.get(img1)[0]
            faces2 = self.app.get(img2)

            result = img2.copy()
            for i in range(min(num_faces, len(faces2))):
                result = self.swap_face(img1, result, face1, faces2[i])
                self.progress = (i + 1) / min(num_faces, len(faces2)) * 90

            result = self.enhance_image(result)
            self.progress = 100
            self.result = result
            self.result_id = str(uuid.uuid4())
        finally:
            self.processing = False
            # ลบไฟล์ที่อัปโหลดมา
            os.remove(img1_path)
            os.remove(img2_path)

face_swap_app = FaceSwapApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    num_faces = int(request.form.get('num_faces', 1))
    recipient_email = request.form.get('recipient_email', '')

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file1 and file2:
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)

        thread = threading.Thread(target=face_swap_app.process_images, args=(filepath1, filepath2, num_faces))
        thread.start()

        return jsonify({'message': 'Processing started', 'recipient_email': recipient_email}), 200

@app.route('/progress')
def progress():
    return jsonify({'progress': face_swap_app.progress, 'processing': face_swap_app.processing})

@app.route('/result')
def result():
    if face_swap_app.processing:
        return jsonify({'message': 'Still processing'}), 202
    
    if face_swap_app.result is not None:
        _, buffer = cv2.imencode('.jpg', face_swap_app.result)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_str, 'result_id': face_swap_app.result_id})
    else:
        return jsonify({'error': 'Result not found'}), 404

@app.route('/download/<result_id>')
def download(result_id):
    if face_swap_app.result is not None and face_swap_app.result_id == result_id:
        _, buffer = cv2.imencode('.jpg', face_swap_app.result)
        try:
            return send_file(
                io.BytesIO(buffer),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='face_swap_result.jpg'
            )
        except TypeError:
            return send_file(
                io.BytesIO(buffer),
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename='face_swap_result.jpg'
            )
    else:
        return jsonify({'error': 'Result not found'}), 404

@app.route('/send_email', methods=['POST'])
def send_email():
    recipient_email = request.json.get('recipient_email')
    if not recipient_email:
        return jsonify({'error': 'Recipient email is required'}), 400

    if face_swap_app.result is not None:
        _, buffer = cv2.imencode('.jpg', face_swap_app.result)
        img_data = buffer.tobytes()

        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = recipient_email
            
            # แยกชื่อผู้ใช้จากอีเมล
            recipient_name = recipient_email.split('@')[0].split('-')[0]
            msg['Subject'] = f"รูปภาพคุณ {recipient_name}"

            text = MIMEText(f"เรียนคุณ {recipient_name}\n\nนี่คือรูปภาพของคุณครับ")
            msg.attach(text)

            image = MIMEImage(img_data, name="face_swap_result.jpg")
            msg.attach(image)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()

            return jsonify({'message': 'Email sent successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to send email: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Result not found'}), 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', debug=True)