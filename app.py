import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import time
import base64
import io
from keras.models import load_model

# Cấu hình giao diện trang
st.set_page_config(
    page_title="MangoLeaf AI | Phân loại bệnh trên lá xoài",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': "https://example.com/bug",
        'About': "# MangoLeaf AI - Hệ thống phát hiện bệnh tiên tiến"
    }
)

# Thông tin bệnhs
DISEASE_INFO = {
    "Anthracnose": {
        "emoji": "🍄",
        "description": "Bệnh nấm gây ra các đốm đen và thối rữa.",
        "symptoms": "Tổn thương sẫm màu, lõm xuống trên lá, thân, hoa và quả.",
        "treatment": "Sử dụng thuốc diệt nấm chứa đồng hoặc lưu huỳnh. Loại bỏ và tiêu hủy các bộ phận cây bị nhiễm bệnh.",
        "prevention": "Đảm bảo khoảng cách thích hợp giữa các cây để lưu thông không khí. Tránh tưới nước từ trên cao.",
        "severity": "Cao"
    },
    "Bacterial Canker": {
        "emoji": "🦠",
        "description": "Nhiễm khuẩn gây tổn thương và rụng lá.",
        "symptoms": "Tổn thương ướt nước chuyển nâu và nứt, chảy dịch vi khuẩn.",
        "treatment": "Cắt tỉa cành bị nhiễm bệnh. Sử dụng thuốc diệt khuẩn gốc đồng.",
        "prevention": "Khử trùng dụng cụ cắt tỉa. Tránh gây vết thương cho cây.",
        "severity": "Trung bình-Cao"
    },
    "Cutting Weevil": {
        "emoji": "🐛",
        "description": "Sâu hại dẫn đến các vết cắt không đều trên lá.",
        "symptoms": "Rìa lá bị khía do bọ cánh cứng trưởng thành ăn.",
        "treatment": "Sử dụng thuốc trừ sâu hoặc biện pháp kiểm soát sinh học như tuyến trùng.",
        "prevention": "Loại bỏ lớp phủ lá nơi bọ cánh cứng trú đông.",
        "severity": "Trung bình"
    },
    "Die Back": {
        "emoji": "🍂",
        "description": "Khô cành và lá từ ngọn xuống gốc.",
        "symptoms": "Cành khô từ ngọn, lá chuyển nâu và rụng.",
        "treatment": "Cắt tỉa cành bị ảnh hưởng và sử dụng thuốc diệt nấm.",
        "prevention": "Duy trì sức sống cây với dinh dưỡng phù hợp.",
        "severity": "Cao"
    },
    "Gall Midge": {
        "emoji": "🪰",
        "description": "Sâu hại gây u bướu dẫn đến cong queo và sưng phồng.",
        "symptoms": "Lá sưng, biến dạng với ấu trùng nhỏ bên trong.",
        "treatment": "Loại bỏ lá bị ảnh hưởng. Sử dụng thuốc trừ sâu phù hợp.",
        "prevention": "Theo dõi các dấu hiệu xâm nhiễm sớm.",
        "severity": "Trung bình"
    },
    "Healthy": {
        "emoji": "✅",
        "description": "Lá không có dấu hiệu bệnh.",
        "symptoms": "Màu xanh bình thường, hình dạng đồng đều, không có đốm hoặc biến dạng.",
        "treatment": "Duy trì các biện pháp canh tác tốt để ngăn ngừa bệnh.",
        "prevention": "Kiểm tra thường xuyên và chăm sóc đúng cách.",
        "severity": "Không"
    },
    "Powdery Mildew": {
        "emoji": "❄️",
        "description": "Nhiễm nấm với các mảng trắng như bột.",
        "symptoms": "Lớp phủ màu trắng như bột trên lá và chồi.",
        "treatment": "Sử dụng thuốc diệt nấm gốc lưu huỳnh hoặc kali bicacbonat.",
        "prevention": "Cải thiện lưu thông không khí xung quanh cây.",
        "severity": "Trung bình"
    },
    "Sooty Mould": {
        "emoji": "🖤",
        "description": "Nấm mốc đen thường do côn trùng hút nhựa.",
        "symptoms": "Lớp phủ màu đen, bồ hóng trên lá có thể lau sạch.",
        "treatment": "Kiểm soát côn trùng tiết mật ngọt. Rửa lá bằng dung dịch xà phòng nhẹ.",
        "prevention": "Quản lý rệp, vảy và các loài gây hại hút nhựa khác.",
        "severity": "Thấp-Trung bình"
    }
}

CLASSES = list(DISEASE_INFO.keys())

# CSS tùy chỉnh
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }

    body {
        background-color: #f8fbf8;
    }

    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f8fbf9 0%, #e6f3ec 100%);
        padding: 0 2rem;
    }

    /* Header styling */
    .header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e6b45;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem 0;
        position: relative;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #1e6b45 0%, #2e8b57 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header:after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 4px;
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
        border-radius: 2px;
    }

    /* Subheader styling */
    .subheader {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2a6840;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.7rem;
        position: relative;
    }

    .subheader:after {
        content: "";
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
        border-radius: 2px;
    }

    /* Card styling */
    .disease-card {
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        background: white;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-left: 4px solid #2e8b57;
    }

    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
    }

    /* Confidence indicators */
    .confidence-high {
        color: #1e6b45;
        font-weight: 700;
        background: rgba(46, 139, 87, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }

    .confidence-medium {
        color: #d97c00;
        font-weight: 700;
        background: rgba(255, 165, 0, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }

    .confidence-low {
        color: #cc3300;
        font-weight: 700;
        background: rgba(255, 69, 0, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }

    /* Button styling */
    .stButton>button {
        border: none;
        background: linear-gradient(90deg, #1e6b45 0%, #2e8b57 100%);
        color: white;
        padding: 0.7rem 1.8rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30, 107, 69, 0.2);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(30, 107, 69, 0.3);
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
    }

    /* File uploader styling */
    .stFileUploader>div>div>div>div {
        border: 2px dashed #2e8b57;
        border-radius: 12px;
        background: rgba(46, 139, 87, 0.05);
        padding: 2rem 1rem;
    }

    /* Severity badges */
    .severity-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff3333 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .severity-medium {
        background: linear-gradient(135deg, #ffd166 0%, #ffb347 100%);
        color: #333;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .severity-low {
        background: linear-gradient(135deg, #06d6a0 0%, #05a181 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #2e8b57;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #1e6b45;
        color: #fff;
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        font-weight: 400;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Image preview styling */
    .image-preview {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 2px solid #fff;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fbf9 0%, #e6f3ec 100%);
        box-shadow: 5px 0 15px rgba(0,0,0,0.03);
    }

    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(46, 139, 87, 0.1);
        margin-bottom: 1.5rem;
    }

    .sidebar-header h2 {
        font-size: 1.8rem;
        color: #1e6b45;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .disease-ref {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }

    .disease-ref:hover {
        transform: translateX(3px);
        box-shadow: 0 5px 12px rgba(0,0,0,0.05);
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid rgba(46, 139, 87, 0.1);
        color: #4a8d65;
        font-size: 0.9rem;
    }

    /* Result container */
    .result-container {
        background: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header {
            font-size: 2.2rem;
        }

        .subheader {
            font-size: 1.4rem;
        }
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f0f9f4 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        margin: 0 5px !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e6b45 0%, #2e8b57 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_model_keras():
    """Load the trained Keras model"""
    model = load_model("keras_model.h5", compile=False)
    return model


@st.cache(allow_output_mutation=True)
def load_class_names():
    with open("labels.txt", "r") as f:
        class_names = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    return class_names


def preprocess_image(image, image_size=224):
    """Preprocess the uploaded image for Keras model"""
    image = np.array(image)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=0)
    return image


def predict(image, model):
    predictions = model.predict(image)
    index = np.argmax(predictions[0])
    confidence = predictions[0][index]
    all_probs = predictions[0]
    return confidence, index, all_probs


def severity_badge(severity):
    """Return styled severity badge"""
    severity = severity.lower()
    if "cao" in severity:
        return f'<span class="severity-high">MỨC ĐỘ CAO</span>'
    elif "trung bình" in severity:
        return f'<span class="severity-medium">MỨC ĐỘ TRUNG BÌNH</span>'
    else:
        return f'<span class="severity-low">MỨC ĐỘ THẤP</span>'


def image_to_base64(image):
    """Convert PIL image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_report(disease_name, confidence, disease_info):
    """Tạo báo cáo văn bản để tải về"""
    return f"""
    🍃 BÁO CÁO PHÂN TÍCH BỆNH TRÊN LÁ XOÀI 🍃
    ======================================

    Chẩn đoán: {disease_name} (Độ tin cậy: {confidence:.2f}%)
    Mức độ nghiêm trọng: {disease_info['severity']}

    MÔ TẢ:
    {disease_info['description']}

    TRIỆU CHỨNG:
    {disease_info['symptoms']}

    ĐIỀU TRỊ KHUYẾN NGHỊ:
    {disease_info['treatment']}

    PHƯƠNG PHÁP PHÒNG NGỪA:
    {disease_info['prevention']}

    Được tạo bởi MangoLeaf AI
    {time.strftime("%Y-%m-%d %H:%M:%S")}
    """


def main():
    # Tiêu đề ứng dụng
    st.markdown("""
    <div class="header">
        🍃 Phân loại bệnh lá xoài - MangoLeaf AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size: 1.1rem; color: #4a8d65; margin-bottom: 2rem; text-align: center;">
        Ứng dụng học sâu để nhận diện và chẩn đoán chính xác các bệnh trên lá xoài. 
        Tải ảnh lên bên dưới để nhận phân tích và khuyến nghị điều trị ngay lập tức.
    </p>
    """, unsafe_allow_html=True)

    # Thanh bên
    with st.sidebar:
        # Logo và tiêu đề
        st.markdown("""
        <div class="sidebar-header">
            <h2>🍃 MangoLeaf AI</h2>
            <p>Hệ thống phát hiện bệnh tiên tiến</p>
        </div>
        """, unsafe_allow_html=True)

        # Menu điều hướng
        st.markdown("### Điều hướng")
        menu = ["Trang chủ", "Cách hoạt động", "Thư viện bệnh", "Giới thiệu", "Liên hệ"]
        choice = st.selectbox("", menu, label_visibility="collapsed")

        st.markdown("### Chọn phương thức nhập ảnh")
        input_method = st.radio(
            "",
            ["Tải ảnh lên", "Chụp ảnh bằng camera"],
            label_visibility="collapsed"
        )

        if input_method == "Tải ảnh lên":
            uploaded_file = st.file_uploader(
                "Chọn một ảnh lá xoài...",
                type=["jpg", "jpeg", "png"],
                help="Tải lên ảnh rõ nét của lá xoài để phân tích",
                label_visibility="collapsed"
            )
            camera_file = None
        else:
            camera_file = st.camera_input("Chụp ảnh lá xoài...", label_visibility="collapsed")
            uploaded_file = None

        # Tham khảo nhanh các loại bệnh
        with st.expander("📚 Tham khảo nhanh các bệnh", expanded=True):
            for disease in CLASSES:
                emoji = DISEASE_INFO[disease]["emoji"]
                st.markdown(f"""
                <div class="disease-ref">
                    <div style="margin-bottom: 0.3rem; font-size: 1rem; display: flex; align-items: center;">
                        <span style="font-size: 1.4rem; margin-right: 10px;">{emoji}</span>
                        <strong>{disease}</strong>
                    </div>
                    <div style="font-size: 0.9rem; color: #555;">
                        {DISEASE_INFO[disease]["description"][:55]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Thông tin liên hệ
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.9rem; color: #4a8d65; padding: 1rem 0;">
            <p style="margin-bottom: 0.8rem; font-weight: 600;">Cần hỗ trợ? Liên hệ đội ngũ của chúng tôi</p>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="background: white; border-radius: 10px; padding: 0.7rem; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✉️</div>
                    <div>quangminhvt1701@gmail.com</div>
                </div>
                <div style="background: white; border-radius: 10px; padding: 0.7rem; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📞</div>
                    <div>0933100017</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Cột nội dung chính
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="subheader">📤 Nhập ảnh</div>', unsafe_allow_html=True)

        input_image = None
        if camera_file is not None:
            input_image = Image.open(camera_file)
            source_type = "Camera"
        elif uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            source_type = "Tải lên"
        else:
            source_type = None

        if input_image:
            # Hiển thị ảnh
            st.markdown(f"""
            <div class="image-preview">
                <img src="data:image/png;base64,{image_to_base64(input_image)}" style="width: 100%;"/>
            </div>
            <div style="text-align: center; margin-top: 1rem; color: #4a8d65; font-weight: 500;">
                Ảnh được chụp từ: {source_type}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Placeholder khi chưa có ảnh
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9f4 0%, #e1f3e9 100%); 
                        border: 2px dashed #2e8b57; border-radius: 12px; 
                        height: 350px; display: flex; justify-content: center; align-items: center;
                        margin-bottom: 1.5rem; box-shadow: 0 8px 20px rgba(0,0,0,0.05);">
                <div style="text-align: center; color: #4a8d65;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" 
                         fill="none" stroke="#2e8b57" stroke-width="2" stroke-linecap="round" 
                         stroke-linejoin="round" style="margin-bottom: 1.5rem;">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <h3 style="color: #1e6b45; margin-bottom: 0.5rem;">Tải lên hình ảnh lá xoài</h3>
                    <p>Hỗ trợ định dạng: JPG, JPEG, PNG</p>
                    <p style="margin-top: 1rem; color: #5d9c74; font-size: 0.9rem;">
                        <span class="tooltip">Mẹo cho kết quả tốt nhất
                            <span class="tooltiptext">Sử dụng ảnh rõ nét, chụp lá trên nền trơn</span>
                        </span>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="subheader">🔍 Kết quả phân tích</div>', unsafe_allow_html=True)

        if input_image:
            # Loading animation
            with st.spinner("Đang phân tích hình ảnh..."):
                # Load model và tên lớp
                model = load_model_keras()
                class_names = load_class_names()

                # Tiền xử lý và dự đoán
                processed_image = preprocess_image(input_image)
                confidence, class_idx, all_probs = predict(processed_image, model)
                confidence_percent = confidence * 100
                disease_name = class_names[class_idx]

                # Lấy thông tin bệnh
                disease_info = DISEASE_INFO.get(disease_name, {
                    "emoji": "❓",
                    "description": "Không có thông tin chi tiết",
                    "symptoms": "Không có thông tin chi tiết",
                    "treatment": "Không có thông tin chi tiết",
                    "prevention": "Không có thông tin chi tiết",
                    "severity": "Không xác định"
                })

            # Hiển thị kết quả
            emoji = DISEASE_INFO[disease_name]["emoji"]
            description = DISEASE_INFO[disease_name]["description"]
            symptoms = DISEASE_INFO[disease_name]["symptoms"]
            confidence_color = "confidence-high" if confidence_percent > 85 else (
                "confidence-medium" if confidence_percent > 60 else "confidence-low")

            with st.container():
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                    <span style="font-size: 2.5rem; margin-right: 15px;">{emoji}</span>
                    <div>
                        <h3 style="color: #1e6b45; margin: 0;">{disease_name}</h3>
                        <p style="margin: 0; color: #666; font-size: 1rem;">{severity_badge(disease_info['severity'])}</p>
                    </div>
                </div>

                <div style="margin-bottom: 1.5rem;">
                    <p><strong style="color: #1e6b45;">📝 Mô tả:</strong> {description}</p>
                    <p><strong style="color: #1e6b45;">🩺 Triệu chứng:</strong> {symptoms}</p>
                    <p><strong style="color: #1e6b45;">📊 Độ tin cậy:</strong> <span class="{confidence_color}">{confidence_percent:.2f}%</span></p>
                </div>

                <div style="margin-bottom: 1rem;">
                    <p><strong style="color: #1e6b45;">📈 Xác suất các bệnh:</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(confidence_percent))

            # Tab điều trị và phòng ngừa
            tab1, tab2 = st.tabs(["💊 Khuyến nghị điều trị", "🛡️ Chiến lược phòng ngừa"])

            with tab1:
                st.markdown(f"""
                <div class="disease-card">
                    <h4 style="margin-top: 0; color: #1e6b45;">Phương pháp điều trị</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6;">{DISEASE_INFO[disease_name]["treatment"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                st.markdown(f"""
                <div class="disease-card">
                    <h4 style="margin-top: 0; color: #1e6b45;">Biện pháp phòng ngừa</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6;">{DISEASE_INFO[disease_name]["prevention"]}</p>
                </div>
                """, unsafe_allow_html=True)

            # Nút tải báo cáo
            st.download_button(
                label="📥 Tải xuống Báo cáo phân tích đầy đủ",
                data=generate_report(disease_name, confidence_percent, DISEASE_INFO[disease_name]),
                file_name=f"bao_cao_{disease_name.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )
        else:
            # Placeholder khi chưa có ảnh
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9f4 0%, #e1f3e9 100%); 
                        border-radius: 12px; padding: 2.5rem; text-align: center; color: #4a8d65;
                        box-shadow: 0 8px 20px rgba(0,0,0,0.05); height: 500px; display: flex; 
                        flex-direction: column; justify-content: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 24 24" 
                     fill="none" stroke="#2e8b57" stroke-width="2" stroke-linecap="round" 
                     stroke-linejoin="round" style="margin-bottom: 1.5rem;">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <h3 style="color: #1e6b45; margin-bottom: 0.8rem; font-size: 1.8rem;">Sẵn sàng để phân tích</h3>
                <p style="font-size: 1.1rem; max-width: 400px; margin: 0 auto 1.5rem;">
                    Tải lên hình ảnh lá xoài để có được phân tích chi tiết và khuyến nghị
                </p>
                <div style="display: inline-block; background: white; padding: 0.8rem 1.5rem; 
                            border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); 
                            color: #2e8b57; font-weight: 600;">
                    ⬅️ Tải lên hình ảnh bằng thanh bên
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 MangoLeaf AI | Hệ thống phát hiện bệnh thực vật tiên tiến
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
