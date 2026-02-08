# Dataset Card: Hand Gesture Recognition Dataset

## Dataset Description

**Dataset Name:** Hand Gesture Recognition Dataset  
**Version:** User-provided  
**Task:** Static hand gesture classification  
**Domain:** Sign Language (American Sign Language gestures)

### Dataset Summary

This dataset contains images of hand gestures for training gesture recognition models. The dataset is organized by gesture class and processed using MediaPipe Hands for landmark extraction.

## Dataset Structure

### Directory Structure
```
TrainingData/
├── 0/          # Class 0 (e.g., 'A')
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 1/          # Class 1 (e.g., 'B')
│   └── ...
└── ...
```

### Data Instances

Each instance consists of:
- **Image:** RGB image containing a hand gesture
- **Label:** Class ID (0-25 for A-Z)
- **Landmarks:** 21 hand landmarks (x, y, z) extracted by MediaPipe

### Data Fields

**Raw Image:**
- Format: JPEG/PNG
- Resolution: Variable (recommended 640x480 or higher)
- Channels: 3 (RGB)

**Processed Landmarks:**
- Shape: (21, 3) for 3D or (21, 2) for 2D
- Features: [x, y, z] normalized coordinates
- Normalization: Min-max, centered on wrist

**Labels:**
- Type: Integer
- Range: 0-25
- Mapping: {0: 'A', 1: 'B', ..., 25: 'Z'}

## Dataset Creation

### Source Data

#### Initial Data Collection
- **Method:** Camera capture (webcam or smartphone)
- **Environment:** Indoor/outdoor with varied lighting
- **Participants:** Multiple users (diversity recommended)
- **Sessions:** Multiple sessions per user

#### Collection Guidelines
1. Clear hand visibility
2. Varied backgrounds
3. Different lighting conditions
4. Multiple hand orientations
5. Consistent gesture formation

### Annotations

#### Annotation Process
- Automatic labeling based on directory structure
- Manual verification recommended
- Quality control for landmark detection

#### Annotation Quality
- MediaPipe confidence threshold: >0.5
- Manual review of failed detections
- Removal of poor-quality samples

### Personal and Sensitive Information

**Privacy Considerations:**
- Images may contain identifiable hand features
- Background may contain personal information
- No facial data should be included
- Metadata scrubbing recommended

**Data Protection:**
- Local storage only
- No cloud uploads (privacy-by-default)
- DVC for version control (optional)
- Encryption for sensitive datasets

## Dataset Statistics

### Size
- **Recommended minimum:** 100 images per class
- **Typical:** 500-1000 images per class
- **Storage:** ~50-500 MB depending on size

### Class Distribution
- **Balanced:** Equal samples per class recommended
- **Imbalanced:** May require weighted sampling

### Quality Metrics
- Landmark detection success rate: >95%
- Image clarity: No excessive blur
- Gesture consistency: Within-class variation analysis

## Considerations for Using the Data

### Social Impact

#### Positive
- Enables accessible communication tools
- Educational applications for sign language
- Improves human-computer interaction

#### Negative
- Potential for misuse in surveillance
- Bias if training data not diverse
- Cultural appropriation concerns

### Biases

#### Known Biases
1. **Skin Tone:** May underrepresent darker skin tones
2. **Hand Size:** Bias toward average adult hands
3. **Background:** Bias toward simple backgrounds
4. **Lighting:** Indoor lighting bias

#### Mitigation Strategies
- Diverse participant recruitment
- Varied environmental conditions
- Data augmentation
- Regular bias audits

### Limitations

1. **Static Gestures Only:** No temporal/dynamic gestures
2. **Single Hand:** Limited multi-hand interactions
3. **ASL Specific:** May not generalize to other sign languages
4. **Cultural Context:** ASL alphabet only

## Additional Information

### Dataset Curators
- User-collected (instructions in repository)
- MediaPipe for automatic landmark extraction

### Licensing
- User datasets: User-defined
- Sample datasets (if provided): MIT License

### Citation
```bibtex
@misc{hand-gesture-dataset-2024,
  title={Hand Gesture Recognition Dataset},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/R-Priyadarshi/handGestureRecognition}
}
```

### Contributions
Guidelines for contributing to the dataset:

1. **Data Quality**
   - High-resolution images (640x480 minimum)
   - Clear hand visibility
   - Proper lighting
   - Varied backgrounds

2. **Diversity**
   - Multiple participants
   - Various skin tones
   - Different hand sizes
   - Age diversity

3. **Format**
   - JPEG or PNG
   - Organized by class
   - Consistent naming

4. **Privacy**
   - No facial images
   - Remove metadata
   - Get participant consent

### Contact
For dataset questions or contributions:
- GitHub Issues: https://github.com/R-Priyadarshi/handGestureRecognition/issues
- Primary Contact: R-Priyadarshi

### Changelog

**v2.0 (2024)**
- Enhanced data collection guidelines
- Privacy-by-default approach
- DVC integration for versioning
- Improved quality control

**v1.x (Previous)**
- Basic image collection
- Simple directory structure

## Usage Guidelines

### Recommended Practices
1. Collect at least 100 samples per class
2. Use multiple participants for diversity
3. Vary lighting and backgrounds
4. Perform quality checks
5. Split into train/val/test sets (80/10/10)

### Data Augmentation
- Rotation: ±15 degrees
- Scale: 0.9-1.1x
- Translation: ±10% of image size
- Brightness: ±20%
- Contrast: ±20%
- Background replacement (optional)

### Evaluation
- Per-class accuracy
- Confusion matrix analysis
- Cross-validation recommended
- Test on held-out participants

## Legal and Ethical

### Consent
- Obtain informed consent from participants
- Explain data usage and storage
- Provide opt-out mechanisms

### Compliance
- GDPR compliance (if applicable)
- Local privacy laws
- Institutional review (if research)

### Responsible Use
- Do not use for unauthorized surveillance
- Respect cultural significance of gestures
- Provide attribution
- Share improvements with community
