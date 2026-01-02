# 🔍 YOLOv8 Object Detection Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, professional web application for real-time object detection powered by YOLOv8 and Streamlit. Detect, identify, and analyze objects in images and videos with state-of-the-art AI technology.

![App Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=YOLOv8+Object+Detector)

## ✨ Features

### 🎯 Core Functionality
- **Real-time Object Detection** - Detect 80 different object classes instantly
- **Single Image Analysis** - Upload and analyze individual images
- **Batch Processing** - Process multiple images at once
- **Video Analysis** - Frame-by-frame object detection in videos
- **Customizable Detection** - Adjust confidence thresholds and visualization options

### 🎨 Modern UI/UX
- **Professional Design** - Sleek gradient-based interface
- **Responsive Layout** - Works on desktop and mobile devices
- **Interactive Controls** - Real-time parameter adjustments
- **Dark Sidebar** - Elegant dark theme sidebar
- **Smooth Animations** - Modern hover effects and transitions

### 🛠️ Technical Features
- **Multiple Model Sizes** - Choose between Nano, Small, and Medium models
- **Flexible Visualization** - Toggle bounding boxes, circles, labels, and confidence scores
- **Download Results** - Export detected images as PNG files
- **Progress Tracking** - Real-time processing status for batch operations
- **Efficient Processing** - Optimized for speed and accuracy

## 🎯 Detectable Objects (80 Classes)

The app can detect the following categories:

| Category | Objects |
|----------|---------|
| **🚗 Vehicles** | car, truck, bus, motorcycle, bicycle, train, boat, airplane |
| **👤 People & Animals** | person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, bird |
| **🍕 Food** | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| **💼 Objects** | backpack, umbrella, handbag, tie, suitcase, bottle, wine glass, cup, fork, knife, spoon, bowl |
| **🏠 Furniture** | chair, couch, potted plant, bed, dining table, toilet |
| **💻 Electronics** | tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator |
| **⚽ Sports** | frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/yolov8-detection-app.git
   cd yolov8-detection-app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📦 Project Structure

```
yolov8-detection-app/
│
├── .streamlit/
│   └── config.toml          # Streamlit configuration
│
├── app.py                   # Main application file
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
└── assets/                 # (Optional) Screenshots and media
    └── screenshots/
```

## 🔧 Configuration

### Model Settings

The app supports three YOLOv8 model sizes:

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| **Nano** | ⚡⚡⚡ | ⭐⭐ | ~6MB | Real-time, mobile |
| **Small** | ⚡⚡ | ⭐⭐⭐ | ~22MB | Balanced performance |
| **Medium** | ⚡ | ⭐⭐⭐⭐ | ~52MB | High accuracy |

### Detection Parameters

- **Confidence Threshold** (0.0 - 1.0)
  - Lower values: More detections, higher false positives
  - Higher values: Fewer detections, higher precision
  - Recommended: 0.25 for general use

### Visualization Options

- **Bounding Boxes** - Rectangular boxes around objects
- **Circles** - Circular highlights around objects
- **Labels** - Object class names
- **Confidence Scores** - Detection confidence percentages
- **Line Thickness** - Adjust visual prominence (1-10)
- **Font Size** - Label text size (0.3-2.0)

## 💻 Usage Guide

### Single Image Detection

1. Navigate to the **"📸 Single Image"** tab
2. Click **"Browse files"** or drag and drop an image
3. View results instantly with detected objects highlighted
4. Click **"💾 Download Result"** to save the annotated image

### Batch Processing

1. Go to the **"📁 Batch Processing"** tab
2. Upload multiple images at once
3. Click **"🚀 Process All Images"**
4. View results for each image with detection summaries

### Video Analysis

1. Switch to the **"🎥 Video Analysis"** tab
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Adjust processing settings:
   - **Process every N frames**: Skip frames for faster processing
   - **Max frames**: Limit total frames analyzed
4. Click **"🎬 Start Analysis"**
5. Watch real-time detection on video frames

## 🌐 Deployment

### Deploy on Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository
   - Click **"Deploy"**

3. **Share your app**
   - Get your public URL: `https://your-app.streamlit.app`
   - Share with the world! 🎉

### Alternative Deployment Options

- **Hugging Face Spaces** - [huggingface.co/spaces](https://huggingface.co/spaces)
- **Render** - [render.com](https://render.com)
- **Railway** - [railway.app](https://railway.app)

## 🛠️ Technologies Used

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Pillow](https://python-pillow.org/)** - Image processing
- **[NumPy](https://numpy.org/)** - Numerical computing

## 📊 Performance

### Speed Benchmarks (Single Image, 640x640)

| Model | GPU (RTX 3060) | CPU (Intel i7) |
|-------|----------------|----------------|
| Nano | ~5ms | ~45ms |
| Small | ~8ms | ~80ms |
| Medium | ~15ms | ~150ms |

*Note: Actual performance varies based on hardware and image complexity*

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions

- [ ] Add custom model training support
- [ ] Implement object tracking in videos
- [ ] Add export to JSON/CSV functionality
- [ ] Create mobile-responsive improvements
- [ ] Add support for webcam input
- [ ] Implement custom class filtering

## 🐛 Known Issues & Limitations

- **First Load**: Model downloads on first run (~6-52MB depending on size)
- **Video Processing**: Large videos may require significant processing time
- **Memory**: Processing many large images simultaneously may cause memory issues
- **Browser Support**: Works best on Chrome, Firefox, and Edge

## 📝 Changelog

### Version 1.0.0 (2024-01-03)
- ✨ Initial release
- 🎨 Modern gradient UI design
- 📸 Single image detection
- 📁 Batch processing support
- 🎥 Video analysis feature
- ⚙️ Customizable detection parameters

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLOv8 implementation
- **Streamlit** team for the intuitive web framework
- **COCO Dataset** for training data and class definitions
- The open-source community for continuous support

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/YOUR_USERNAME/yolov8-detection-app](https://github.com/YOUR_USERNAME/yolov8-detection-app)

Live Demo: [https://your-app.streamlit.app](https://your-app.streamlit.app)

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/yolov8-detection-app&type=Date)](https://star-history.com/#YOUR_USERNAME/yolov8-detection-app&Date)

---

<div align="center">

**Made with ❤️ using YOLOv8 and Streamlit**

[Report Bug](https://github.com/YOUR_USERNAME/yolov8-detection-app/issues) · [Request Feature](https://github.com/YOUR_USERNAME/yolov8-detection-app/issues) · [Documentation](https://github.com/YOUR_USERNAME/yolov8-detection-app/wiki)

</div>
