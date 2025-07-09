// START OF FILE server.js
const express = require('express');
// const { PythonShell } = require('python-shell'); // <-- XÓA DÒNG NÀY
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const multer = require('multer');
const fs = require('fs');
const axios = require('axios'); // <-- THÊM DÒNG NÀY

const app = express();
const port = 3001;

// --- Configuration ---
const UPLOADS_DIR = 'public/uploads/';
const SCRIPTS_DIR = 'scripts'; // Giờ không dùng trực tiếp nhưng giữ lại nếu có script khác
// const PYTHON_EXECUTABLE = '...'; // <-- XÓA HOẶC COMMENT DÒNG NÀY
const PYTHON_API_URL = 'http://127.0.0.1:8000/predict_image/'; // <-- THÊM URL API Python

// --- Ensure directories exist ---
const uploadsFullPath = path.join(__dirname, UPLOADS_DIR);
// const scriptsFullPath = path.join(__dirname, SCRIPTS_DIR); // Không cần check nữa

if (!fs.existsSync(uploadsFullPath)) {
    console.log(`Creating directory: ${uploadsFullPath}`);
    fs.mkdirSync(uploadsFullPath, { recursive: true });
}
// Bỏ check PYTHON_EXECUTABLE và scriptsFullPath nếu không còn dùng

// --- Multer, Middleware, Static Files, '/' route (Giữ nguyên) ---
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsFullPath);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file) {
        console.error('Upload Error: No file received.');
        return res.status(400).json({ error: 'No image file uploaded' });
    }
    const relativeImagePath = `${UPLOADS_DIR.replace('public/', '')}${req.file.filename}`;
    console.log(`Image uploaded successfully: ${req.file.filename}, Path: ${relativeImagePath}`);
    res.json({ imagePath: relativeImagePath });
});


// --- Route để xử lý dự đoán (ĐÃ SỬA ĐỂ GỌI API PYTHON) ---
app.post('/predict', async (req, res) => { // <-- Thêm async
    const relativeImagePath = req.body.imagePath;
    if (!relativeImagePath) {
        console.error('Predict Error: No imagePath in request body.');
        return res.status(400).json({ error: 'No image path provided in request body' });
    }

    const fullImagePath = path.join(__dirname, 'public', relativeImagePath);
    console.log(`Sending image path to Python API: ${fullImagePath}`);

    if (!fs.existsSync(fullImagePath)) {
         console.error(`Predict Error: Image file not found locally at ${fullImagePath}`);
         // Vẫn cần kiểm tra ở đây trước khi gửi đi
         return res.status(404).json({ error: 'Image file not found on server before sending to API' });
    }

    try {
        // Gửi yêu cầu POST đến API Python bằng axios
        const response = await axios.post(PYTHON_API_URL, {
            image_path: fullImagePath // Gửi đường dẫn tuyệt đối
        });

        // Kiểm tra kết quả từ API Python
        if (response.data) {
            console.log('Prediction successful from Python API:', response.data.prediction);
            res.json(response.data); // Gửi kết quả về frontend
        } else {
             // Trường hợp API Python trả về không có dữ liệu (ít xảy ra với FastAPI)
             console.error('Error: No data received from Python API');
             res.status(500).json({ error: 'Prediction service returned empty data' });
        }

    } catch (error) {
        // Xử lý lỗi khi gọi API Python
        console.error('Error calling Python API:', error.message);
        if (error.response) {
            // Lỗi trả về từ API Python (ví dụ: 404, 500, 503)
            console.error('Python API Error Status:', error.response.status);
            console.error('Python API Error Data:', error.response.data);
            // Gửi lại lỗi từ API Python cho frontend
            res.status(error.response.status).json({
                error: 'Prediction service failed',
                details: error.response.data.detail || 'Unknown error from prediction service'
            });
        } else if (error.request) {
            // Không kết nối được đến API Python
            console.error('Error: Could not connect to Python API at', PYTHON_API_URL);
            res.status(503).json({ // Service Unavailable
                error: 'Prediction service unavailable',
                details: `Could not connect to ${PYTHON_API_URL}. Is the Python API server running?`
            });
        } else {
            // Lỗi khác trong quá trình thiết lập yêu cầu
            console.error('Axios setup error:', error.message);
            res.status(500).json({ error: 'Failed to send prediction request', details: error.message });
        }
    }
});


// --- Khởi động server Node.js ---
app.listen(port, () => {
    console.log(`Node.js Server running at http://localhost:${port}`);
    console.log(`Serving static files from: ${path.join(__dirname, 'public')}`);
    console.log(`Uploading images to: ${uploadsFullPath}`);
    console.log(`Forwarding predictions to Python API at: ${PYTHON_API_URL}`);
    // console.log(`Using Python executable: ...`); // <-- XÓA HOẶC COMMENT
    // console.log(`Expecting Python script in: ...`); // <-- XÓA HOẶC COMMENT
});
// END OF FILE server.js