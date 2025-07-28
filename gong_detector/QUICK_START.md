# Quick Start: Gong Detection System

## **Core Commands**

### **Basic Detection:**
```bash
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID"
```

### **With Trained Classifier (Enhanced):**
```bash
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --use_version_one
```

### **Bulk Processing:**
```bash
python -m gong_detector.core.bulk_process --version_one
```

### **Sample Collection:**
```bash
# Collect positive samples for training
python -m gong_detector.core.detect_from_youtube "URL" --save_positive_samples

# Interactive manual collection
python -m gong_detector.core.manual_sample_collector

# Collect negative samples
python -m gong_detector.core.negative_sample_collector "URL"
```

## **Project Structure**

### **Core Detection (`/core`)**
- `yamnet_runner.py` - YAMNet model integration with trained classifier support
- `detect_from_youtube.py` - Single video detection with optimized batch processing
- `bulk_process.py` - Multi-video processing with performance optimizations
- `manual_sample_collector.py` - Interactive training sample collection
- `negative_sample_collector.py` - Non-gong sample collection for training
- `models/` - Trained classifier files (classifier.pkl, config.json)

### **Training Pipeline (`/training`)**
- `scripts/extract_embeddings.py` - Extract YAMNet features from audio samples
- `scripts/train_classifier.py` - Train Random Forest on embeddings
- `scripts/evaluate_model.py` - Evaluate trained model performance
- `data/validated_samples/` - Curated training data (positive/negative)

### **Performance Optimizations**
- **Batch Processing**: 5000 embeddings per batch (configurable)
- **Multi-threading**: 8 inter-op + 4 intra-op threads
- **Memory Efficient**: Optimized for 16GB+ systems
- **Conservative Defaults**: 0.925 threshold for high precision

## **Key Features**

- **Trained Classifier**: 99.3% accuracy on validation data
- **Batch Processing**: 5-10x faster than sequential processing
- **Conservative Detection**: High thresholds prevent false positives
- **Training Integration**: Seamless workflow from detection to model training
- **Performance Monitoring**: Real-time feedback on resource usage