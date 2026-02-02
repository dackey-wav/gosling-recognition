from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms, models
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import io
import os
import base64

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_gosling_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
print(f"[INFO] Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# Face detector (runs on CPU to avoid torchvision CUDA compatibility issues)
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device='cpu'  # Force CPU for face detection
)
print("[INFO] Face detector (MTCNN) loaded on CPU")

# Classification model (ResNet18)
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[INFO] Model loaded from: {MODEL_PATH}")

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- FASTAPI APP ---
app = FastAPI(
    title="Gosling Recognition API",
    description="Upload a photo to check if it's Ryan Gosling",
    version="1.0.0"
)

# --- HTML INTERFACE ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ryan Gosling Detector</title>
    <style>
        * {
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px 20px;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            background: #f0f0ff;
            border-color: #764ba2;
        }
        .upload-area.dragover {
            background: #e0e0ff;
            border-color: #764ba2;
        }
        input[type="file"] {
            display: none;
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        .preview-container {
            display: none;
            margin: 20px 0;
        }
        .preview-container img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 15px;
            display: none;
        }
        .result.success {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        .result.failure {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        .result.error {
            background: #fff3cd;
            border: 2px solid #ffc107;
        }
        .result-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        .result-text {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 14px;
            color: #666;
        }
        .loading {
            display: none;
            margin-top: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .face-preview {
            margin-top: 15px;
        }
        .face-preview img {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 3px solid #667eea;
        }
        .face-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Ryan Gosling Detector</h1>
        <p class="subtitle">Upload a photo to check if it's Ryan Gosling</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <p>Click or drag & drop an image here</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="preview-container" id="previewContainer">
            <img id="preview" src="" alt="Preview">
        </div>
        
        <button id="analyzeBtn" disabled>🔍 Analyze</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-icon" id="resultIcon"></div>
            <div class="result-text" id="resultText"></div>
            <div class="confidence" id="confidence"></div>
            <div class="face-preview" id="facePreview"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const preview = document.getElementById('preview');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        
        let selectedFile = null;

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }
            
            selectedFile = file;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
                analyzeBtn.disabled = false;
                result.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        // Analyze button
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (data.error) {
                    result.className = 'result error';
                    document.getElementById('resultIcon').textContent = '⚠️';
                    document.getElementById('resultText').textContent = data.error;
                    document.getElementById('confidence').textContent = '';
                    document.getElementById('facePreview').innerHTML = '';
                } else if (data.is_gosling) {
                    result.className = 'result success';
                    document.getElementById('resultIcon').textContent = '✅';
                    document.getElementById('resultText').textContent = "It's Ryan Gosling!";
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                    showFacePreview(data.face_image);
                } else {
                    result.className = 'result failure';
                    document.getElementById('resultIcon').textContent = '❌';
                    document.getElementById('resultText').textContent = "Not Ryan Gosling";
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                    showFacePreview(data.face_image);
                }
            } catch (error) {
                loading.style.display = 'none';
                result.style.display = 'block';
                result.className = 'result error';
                document.getElementById('resultIcon').textContent = '❌';
                document.getElementById('resultText').textContent = 'Connection error';
                document.getElementById('confidence').textContent = error.message;
                document.getElementById('facePreview').innerHTML = '';
            }
            
            analyzeBtn.disabled = false;
        });
        
        function showFacePreview(base64Image) {
            if (base64Image) {
                document.getElementById('facePreview').innerHTML = `
                    <img src="data:image/jpeg;base64,${base64Image}" alt="Detected face">
                    <div class="face-label">Detected face</div>
                `;
            } else {
                document.getElementById('facePreview').innerHTML = '';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main page with upload interface"""
    return HTML_PAGE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze uploaded image for Ryan Gosling's face.
    
    Returns:
        - is_gosling: bool - whether the face belongs to Ryan Gosling
        - confidence: float - prediction confidence (0.0 - 1.0)
        - face_image: str - base64 encoded detected face (for preview)
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        return {"error": "File must be an image", "is_gosling": False, "confidence": 0.0}
    
    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize if too large
        max_size = 1200
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)))
        
        # Detect face
        face_tensor = mtcnn(image)
        
        if face_tensor is None:
            return {
                "error": "No face detected in the image",
                "is_gosling": False,
                "confidence": 0.0,
                "face_image": None
            }
        
        # Convert face tensor to image for preview
        face_for_preview = face_tensor.permute(1, 2, 0)
        face_for_preview = (face_for_preview + 1) / 2
        face_for_preview = (face_for_preview * 255).clamp(0, 255).byte().cpu().numpy()
        face_pil = Image.fromarray(face_for_preview)
        
        # Encode face to base64 for frontend
        buffer = io.BytesIO()
        face_pil.save(buffer, format="JPEG")
        face_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare face for classification
        face_image = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(face_image)
            prob_negative = torch.sigmoid(output).item()
        
        prob_gosling = 1.0 - prob_negative
        is_gosling = prob_gosling > 0.5
        confidence = prob_gosling if is_gosling else prob_negative
        
        return {
            "is_gosling": is_gosling,
            "confidence": round(confidence, 4),
            "prob_gosling": round(prob_gosling, 4),
            "prob_other": round(prob_negative, 4),
            "face_image": face_base64
        }
        
    except Exception as e:
        return {
            "error": f"Processing error: {str(e)}",
            "is_gosling": False,
            "confidence": 0.0,
            "face_image": None
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting server at http://localhost:8000")
    print("[INFO] Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
