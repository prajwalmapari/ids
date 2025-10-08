# IDS - Face Recognition System

## 🌐 Overview

Advanced **cloud-based** face recognition system using InsightFace for accurate processing with Azure storage integration. The system processes face recognition entirely in the cloud without any local storage dependencies.

## ✨ Key Features

- **🎯 High Accuracy**: Uses InsightFace Buffalo_L model (68-83% confidence rates)
- **☁️ Cloud**: All data stored and processed in Azure cloud
- **👥 Multi-Target**: Detects and recognizes multiple faces simultaneously
- **🔐 Secure**: Environment-based configuration with Azure integration
- **📊 Real-time Logging**: All predictions logged to Azure storage
- **📸 Evidence Collection**: Unauthorized person images automatically stored in Azure



## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/prajwalmapari/ids.git
cd ids

# Setup Python environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure credentials

# Test the system
python main.py --mode image --input group.png
```



## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
AZURE_CONTAINER_NAME=sr001
AZURE_EMBEDDINGS_BLOB=authorised/authorised person/authorized_persons.json
AZURE_LOG_BLOB=unauthorised_person/detection_logs/
AZURE_IMAGES_FOLDER=unauthorised_person/Image/

# Recognition Settings
RECOGNITION_THRESHOLD=0.6
```

### Azure Storage Structure

```
Container: sr001/
├── authorised/authorised person/
│   └── authorized_persons.json                    # Face embeddings database
├── unauthorised_person/
│   ├── detection_logs/
│   │   ├── detection_log_2025-10-08.json         # Daily JSON logs
│   │   ├── detection_log_2025-10-07.json
│   │   └── ...                                    # Daily detection logs
│   └── Image/
│       ├── 2025-10-08/                           # Date-based folders
│       │   ├── unauthorized_20251008_143052_face_1.jpg
│       │   └── unauthorized_20251008_143055_face_2.jpg
│       ├── 2025-10-07/
│       │   └── ...                                # Previous day images
│       └── ...                                    # Other dates
```

### JSON Log Format

Each daily log file contains structured detection data:

```json
{
  "metadata": {
    "last_updated": "2025-10-08T14:30:55.123456",
    "total_detections": 125,
    "unauthorized_count": 78,
    "authorized_count": 47,
    "log_file": "detection_log_2025-10-08.json",
    "azure_container": "sr001"
  },
  "detections": [
    {
      "id": 1,
      "timestamp": "2025-10-08T14:28:45.789012",
      "human_time": "2025-10-08 14:28:45",
      "status": "AUTHORIZED",
      "person_name": "dhruv",
      "confidence": "81%",
      "location": "Camera",
      "alert_level": "LOW",
      "additional_info": null
    }
  ]
}
```

## 🖥️ Usage

### Process Image
```bash
python main.py --mode image --input photo.jpg --output results.json
```

### Process Video
```bash
python main.py --mode video --input video.mp4 --output results.json
```

### Live Webcam
```bash
python main.py --mode webcam
```

## 📊 System Performance

### Test Results (16-person group image):
- **✅ Authorized Personnel**: 9/16 correctly identified
- **❌ Unauthorized Persons**: 7/16 properly flagged
- **🎯 Recognition Accuracy**: 68.4% - 82.5% confidence
- **📸 Face Detection**: 100% success rate
- **☁️ Cloud Storage**: 100% (no local dependencies)

### Recognized Personnel:
- dhruv (81.4%), sagar (82.5%), dwarika (80.0%)
- omkar (79.4%), rishabh (77.0%), rupesh (78.3%)
- nisha (74.1%), bapu (69.2%), vaibhavi (68.4%)


## 📦 Dependencies

- **InsightFace**: State-of-the-art face recognition
- **Azure Storage**: Cloud data management
- **OpenCV**: Computer vision processing
- **NumPy**: Numerical computations
- **Python-dotenv**: Environment configuration



- **InsightFace**: State-of-the-art face recognition

- **Azure Storage**: Cloud data management

- **OpenCV**: Computer vision processing

- **NumPy**: Numerical computation

- **Python-dotenv**: Environment configuration

## Persistent face database (auto-generated)

- **Azure Integration**: Secure cloud authentication

- **No Local Storage**: Zero local data persistence

- **Audit Trail**: Complete logging of all recognition events

- **Access Control**: Employee-level authorization validation


## 🌐 Cloud-First Benefits

1. **Scalability**: Horizontal scaling ready
2. **Reliability**: Azure-backed storage and processing
3. **Security**: No local data exposure
4. **Maintenance**: Cloud-managed infrastructure
### Common Issues

1. **No faces detected**: Check image quality and lighting


2. **Poor recognition**: Adjust threshold or add more reference images

3. **Slow performance**: Reduce detection size or use batch mode


4. **Memory issues**: Process videos in chunks or reduce resolution

5. **Accessibility**: Access from anywhere

6. **Backup**: Automatic Azure redundancy

## 🔍 Output Format

### Recognition Results
```json
{
  "source": "Image: photo.jpg",
  "total_faces": 16,
  "authorized_count": 9,
  "unauthorized_count": 7,
  "faces": [
    {
      "face_id": "face_1",
      "recognition": {
        "name": "dhruv",
        "confidence": 0.814,
        "authorized": true
      },
      "metadata": {
        "employee_id": "person_1757852323",
        "department": "Security",
        "access_level": "MEDIUM"
      },
      "age": 36,
      "gender": "Female"
    }
  ]
}
```

## 🆘 Troubleshooting

### Common Issues

**Azure Connection Failed**
```bash
# Verify connection string
echo $AZURE_STORAGE_CONNECTION_STRING
```

**InsightFace Model Download**
```bash
# Models auto-download to ~/.insightface/models/
# Ensure stable internet connection on first run
```

**No Faces Detected**
```bash
# Check image quality and lighting
# Ensure faces are clearly visible and well-lit
```

## 📞 Support

- **Repository**: https://github.com/prajwalmapari/ids
- **Issues**: Create GitHub issues for bug reports
- **License**: MIT License

---


**System Verified**: 9/16 authorized personnel correctly identified with 68-83% confidence