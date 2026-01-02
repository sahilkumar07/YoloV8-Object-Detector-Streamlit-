import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Page config with modern theme
st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .stApp {
        background: #f8f9fa;
    }
    
    /* Custom header */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .hero-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Success message */
    .success-badge {
        display: inline-block;
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔍 YOLO Object Detector</div>
        <div class="hero-subtitle">AI-Powered Object Detection & Recognition</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar with modern styling
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Model selection with icons
    st.markdown("### 🤖 Model Settings")
    model_size = st.selectbox(
        "Model Size",
        ["⚡ Nano (Fastest)", "🚀 Small (Balanced)", "🎯 Medium (Most Accurate)"],
        index=0
    )
    
    # Map selection to model name
    model_map = {
        "⚡ Nano (Fastest)": "yolov8n.pt",
        "🚀 Small (Balanced)": "yolov8s.pt",
        "🎯 Medium (Most Accurate)": "yolov8m.pt"
    }
    model_name = model_map[model_size]
    
    st.markdown("---")
    
    # Detection parameters
    st.markdown("### 🎚️ Detection Parameters")
    confidence = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.05,
        help="Minimum confidence for detection"
    )
    
    st.markdown("---")
    
    # Visualization options
    st.markdown("### 🎨 Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        draw_boxes = st.checkbox("📦 Boxes", value=True)
        show_labels = st.checkbox("🏷️ Labels", value=True)
    with col2:
        draw_circles = st.checkbox("⭕ Circles", value=False)
        show_confidence = st.checkbox("📊 Scores", value=True)
    
    box_thickness = st.slider("Line Thickness", 1, 10, 3)
    font_size = st.slider("Font Size", 0.3, 2.0, 0.8, 0.1)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("Built with YOLOv8 and Streamlit. Detects 80 object classes in real-time.")

# Load model
@st.cache_resource
def load_yolo_model(model_name):
    try:
        model = YOLO(model_name)
        return model, None
    except Exception as e:
        return None, str(e)

with st.spinner("🔄 Loading AI model..."):
    model, error = load_yolo_model(model_name)

if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()
else:
    st.sidebar.success(f"✅ {model_name} loaded")

# Detection function
def detect_and_annotate(image, conf_threshold):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    results = model(image, conf=conf_threshold, verbose=False)[0]
    annotated_img = image.copy()
    boxes = results.boxes
    detection_data = []
    
    if len(boxes) > 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            detection_data.append({
                'id': idx + 1,
                'name': class_name,
                'confidence': conf,
                'box': (x1, y1, x2, y2)
            })
            
            # Color palette
            colors = {
                'person': (255, 107, 107),
                'car': (78, 205, 196),
                'default': (102, 126, 234)
            }
            color = colors.get(class_name, colors['default'])
            
            if draw_boxes:
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, box_thickness)
            
            if draw_circles:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 2 + 10
                cv2.circle(annotated_img, (center_x, center_y), radius, color, 2)
            
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{conf:.0%}")
            
            label = " ".join(label_parts)
            
            if label:
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2
                )
                
                # Modern label with rounded corners effect
                padding = 8
                cv2.rectangle(
                    annotated_img,
                    (x1, y1 - text_height - padding * 2),
                    (x1 + text_width + padding * 2, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated_img,
                    label,
                    (x1 + padding, y1 - padding),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
    
    return annotated_img, detection_data

# Main tabs
tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Processing", "🎥 Video Analysis"])

# TAB 1: Single Image
with tab1:
    st.markdown('<p class="section-header">Single Image Detection</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="single_image"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 📥 Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels")
        
        with col2:
            st.markdown("#### 🎯 Detection Results")
            
            with st.spinner("🔍 Analyzing image..."):
                annotated_img, detections = detect_and_annotate(image, confidence)
                
                if annotated_img.dtype == np.uint8:
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                else:
                    annotated_img_rgb = annotated_img
                
                st.image(annotated_img_rgb, use_container_width=True)
                
                if detections:
                    st.markdown(f'<div class="success-badge">✅ {len(detections)} objects detected</div>', unsafe_allow_html=True)
                
                # Download button
                result_img = Image.fromarray(annotated_img_rgb)
                import io
                buf = io.BytesIO()
                result_img.save(buf, format='PNG')
                
                st.download_button(
                    label="💾 Download Result",
                    data=buf.getvalue(),
                    file_name="yolo_detection_result.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        # Detection cards
        if detections:
            st.markdown("---")
            st.markdown('<p class="section-header">Detection Details</p>', unsafe_allow_html=True)
            
            cols = st.columns(min(len(detections), 4))
            for idx, det in enumerate(detections):
                with cols[idx % 4]:
                    st.metric(
                        label=f"🎯 Object {det['id']}",
                        value=det['name'].title(),
                        delta=f"{det['confidence']:.1%}"
                    )
            
            # Detailed list in expander
            with st.expander("📋 View Detailed Coordinates"):
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    st.markdown(f"""
                    **{det['id']}.** `{det['name']}` - Confidence: **{det['confidence']:.2%}**  
                    📍 Location: `({x1}, {y1})` → `({x2}, {y2})`
                    """)
        else:
            st.info("💡 No objects detected. Try adjusting the confidence threshold in the sidebar.")

# TAB 2: Batch Processing
with tab2:
    st.markdown('<p class="section-header">Batch Image Processing</p>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch processing",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_images"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images ready for processing")
        
        if st.button("🚀 Process All Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.markdown(f"Processing **{file.name}** ({idx + 1}/{len(uploaded_files)})")
                
                st.markdown("---")
                st.markdown(f"#### 🖼️ {file.name}")
                
                image = Image.open(file)
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                
                with col2:
                    annotated_img, detections = detect_and_annotate(image, confidence)
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, caption="Detected", use_container_width=True)
                
                if detections:
                    objects = ', '.join([d['name'] for d in detections])
                    st.success(f"✅ Found {len(detections)} objects: {objects}")
                else:
                    st.warning("⚠️ No objects detected")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            st.balloons()
            st.success("🎉 All images processed successfully!")

# TAB 3: Video
with tab3:
    st.markdown('<p class="section-header">Video Analysis</p>', unsafe_allow_html=True)
    st.info("💡 Upload a video for frame-by-frame object detection")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 📥 Original Video")
            st.video(uploaded_video)
        
        with col2:
            st.markdown("#### ⚙️ Processing Options")
            process_every_n = st.slider("Process every N frames", 1, 10, 2)
            max_frames = st.number_input("Max frames to process", 10, 500, 100)
        
        if st.button("🎬 Start Analysis", type="primary", use_container_width=True):
            st.markdown("---")
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            progress_bar = st.progress(0)
            status = st.empty()
            
            frame_count = 0
            processed = 0
            detected_objects = set()
            
            while cap.isOpened() and processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % process_every_n == 0:
                    annotated_frame, detections = detect_and_annotate(frame, confidence)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, use_container_width=True)
                    
                    for det in detections:
                        detected_objects.add(det['name'])
                    
                    processed += 1
                    progress_bar.progress(processed / max_frames)
                    status.markdown(f"**Frame {processed}/{max_frames}** | Objects: {', '.join(detected_objects)}")
                
                frame_count += 1
            
            cap.release()
            st.success(f"✅ Analysis complete! Detected objects: {', '.join(detected_objects)}")

# Footer with detectable classes
st.markdown("---")
with st.expander("📋 View All 80 Detectable Object Classes"):
    if model:
        classes = list(model.names.values())
        
        # Group by category
        categories = {
            "🚗 Vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"],
            "👤 People & Animals": ["person", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird"],
            "🍕 Food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
            "💼 Objects": ["backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
            "🏠 Furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
            "💻 Electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
            "⚽ Sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"]
        }
        
        for category, items in categories.items():
            st.markdown(f"**{category}**")
            category_classes = [c for c in classes if c in items]
            if category_classes:
                st.markdown(", ".join(category_classes))
            st.markdown("")

# Modern footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #64748b;'>
        <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
            Powered by YOLOv8 & Streamlit
        </p>
        <p style='font-size: 0.9rem;'>
            Real-time object detection with state-of-the-art AI
        </p>
    </div>
""", unsafe_allow_html=True)