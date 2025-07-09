// START OF FILE script.js

// Lấy các phần tử DOM cần thiết một lần khi script tải
const fileInput = document.getElementById('imageInput');
const filenameDisplay = document.getElementById('filename-display');
const resultDiv = document.getElementById('result');
const imagePreview = document.getElementById('imagePreview'); // Lấy thẻ img xem trước

// Kiểm tra xem các phần tử DOM có tồn tại không
if (!fileInput || !filenameDisplay || !resultDiv || !imagePreview) {
    console.error("CRITICAL: Could not find required DOM elements (input, filename display, result div, or image preview).");
    alert("Page structure error. Please check the HTML.");
} else {
    // Chỉ thêm event listener nếu các phần tử tồn tại
    fileInput.addEventListener('change', function() {
        // Lấy file đầu tiên được chọn
        const file = this.files[0];

        if (file) {
            // Hiển thị tên file
            filenameDisplay.textContent = file.name;

            // Kiểm tra xem file có phải là ảnh không
            if (file.type.startsWith('image/')) {
                // Tạo một đối tượng FileReader để đọc nội dung file
                const reader = new FileReader();

                // Sự kiện này được kích hoạt khi đọc file hoàn tất
                reader.onload = function(e) {
                    // Đặt thuộc tính src của thẻ img thành dữ liệu ảnh đã đọc
                    imagePreview.src = e.target.result;
                    // Hiển thị thẻ img (xóa lớp hidden)
                    imagePreview.classList.remove('hidden');
                }
                // Bắt đầu đọc file dưới dạng Data URL
                reader.readAsDataURL(file);

            } else {
                // Nếu không phải file ảnh, ẩn preview và báo lỗi (tùy chọn)
                filenameDisplay.textContent = 'Please select an image file.';
                imagePreview.src = '#'; // Xóa ảnh cũ
                imagePreview.classList.add('hidden'); // Ẩn đi
            }

            // Xóa kết quả dự đoán cũ khi chọn file mới
            resultDiv.innerHTML = '';
            resultDiv.classList.remove('visible', 'loading', 'error');

        } else {
            // Nếu không chọn file nào (ví dụ: nhấn cancel)
            filenameDisplay.textContent = 'No file chosen';
            imagePreview.src = '#'; // Xóa ảnh cũ
            imagePreview.classList.add('hidden'); // Ẩn đi
             // Xóa kết quả cũ
             resultDiv.innerHTML = '';
             resultDiv.classList.remove('visible', 'loading', 'error');
        }
    });
}

// Hàm xử lý dự đoán khi nhấn nút
async function predict() {
    // Kiểm tra lại file input và result div (phòng trường hợp)
    if (!fileInput || !resultDiv) {
        console.error("Required DOM elements not found for prediction.");
        alert("An error occurred. Cannot proceed with prediction.");
        return;
    }

    // --- Bước 1: Thiết lập trạng thái chờ ban đầu ---
    resultDiv.innerHTML = ''; // Xóa nội dung cũ
    resultDiv.classList.remove('error', 'visible'); // Xóa các lớp trạng thái cũ
    resultDiv.classList.add('loading', 'visible'); // Thêm lớp loading và visible
    resultDiv.textContent = 'Uploading image...'; // Đặt text trạng thái ban đầu

    // --- Bước 2: Kiểm tra xem có file nào được chọn không ---
    if (!fileInput.files || fileInput.files.length === 0) {
        resultDiv.textContent = 'Please select an image file first!';
        resultDiv.classList.remove('loading'); // Xóa loading
        resultDiv.classList.add('error'); // Thêm error
        return;
    }

    // --- Bước 3: Chuẩn bị và gửi yêu cầu Upload ---
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
        // --- Gửi yêu cầu Upload ---
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        // Cập nhật trạng thái chờ
        resultDiv.textContent = 'Image uploaded. Requesting prediction...';

        // --- Xử lý kết quả Upload ---
        if (!uploadResponse.ok) {
            let errorMsg = `Upload failed with status: ${uploadResponse.status}`;
            try {
                const errorData = await uploadResponse.json();
                errorMsg = `Upload failed: ${errorData.error || uploadResponse.statusText}`;
            } catch (jsonError) {
                errorMsg = `Upload failed: ${uploadResponse.statusText || 'Server error'}`;
            }
            throw new Error(errorMsg);
        }

        const uploadData = await uploadResponse.json();
        if (uploadData.error) {
             throw new Error(`Server upload error: ${uploadData.error}`);
        }

        // --- Bước 4: Chuẩn bị và gửi yêu cầu Predict ---
        resultDiv.textContent = 'Processing image... Please wait.'; // Cập nhật trạng thái

        const predictResponse = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imagePath: uploadData.imagePath })
        });

        // Xóa lớp loading KHI có phản hồi
        resultDiv.classList.remove('loading');

        // --- Xử lý kết quả Predict ---
        if (!predictResponse.ok) {
             let errorMsg = `Prediction failed with status: ${predictResponse.status}`;
             try {
                 const errorData = await predictResponse.json();
                 errorMsg = `Prediction failed: ${errorData.details || errorData.error || predictResponse.statusText}`;
             } catch (jsonError) {
                  errorMsg = `Prediction failed: ${predictResponse.statusText || 'Server error'}`;
             }
             throw new Error(errorMsg);
        }

        const predictData = await predictResponse.json();
        if (predictData.error) {
            throw new Error(`Prediction error from service: ${predictData.details || predictData.error}`);
        }

        // --- Bước 5: Hiển thị kết quả thành công ---
        let probabilitiesHtml = '<ul>';
        const sortedProbs = Object.entries(predictData.probabilities)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5);

        for (const [className, probability] of sortedProbs) {
             probabilitiesHtml += `<li>${className}: <span class="probability-value">${(probability * 100).toFixed(2)}%</span></li>`;
        }
        probabilitiesHtml += '</ul>';

        resultDiv.innerHTML = `
            <span class="prediction-title">${predictData.prediction}</span>
            <span class="probabilities-label">Top Probabilities:</span>
            ${probabilitiesHtml}
        `;
        // Đảm bảo lớp visible vẫn còn

    } catch (err) {
        // --- Bước 6: Xử lý bất kỳ lỗi nào xảy ra trong quá trình ---
        console.error("Prediction process failed:", err);
        resultDiv.textContent = `Error: ${err.message || 'An unknown error occurred.'}`;
        resultDiv.classList.remove('loading'); // Đảm bảo xóa loading
        resultDiv.classList.add('error'); // Thêm lớp error
        resultDiv.classList.add('visible'); // Đảm bảo div lỗi hiển thị
    }
}
// END OF FILE script.js