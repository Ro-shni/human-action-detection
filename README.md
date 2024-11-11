# Human Action Detection Using Deep Learning 📹

## Overview
This project aims to develop a Human Action Detection system using deep learning techniques. The model can detect various actions such as Punch, PushUps, and everyday activities like WalkingWithDog and Biking from video sequences. It leverages CNN (Convolutional Neural Networks) for feature extraction and LSTM (Long Short-Term Memory) for temporal sequence learning.

## Project Goal
To create a reliable system that can automatically identify specific human actions in real-time, offering potential use in surveillance and monitoring applications.

## Key Features
- Detects and classifies human actions from video sequences.
- Achieves an accuracy of 91.16% on the validation dataset.
- Supports real-time action detection using video feed or pre-recorded videos.
- Deployed using Streamlit for a user-friendly interface.

## Dataset
The dataset used includes various human action videos categorized into four main classes:
- **WalkingWithDog**
- **Punch**
- **PushUps**
- **Biking**

## Data Preprocessing
- Video frames are resized and normalized.
- One-hot encoding is applied to the labels for multi-class classification.
- Frames are fed into a CNN + LSTM model for feature extraction and temporal analysis.

## Model Architecture
The model combines:
- **Convolutional Neural Network (CNN):** For spatial feature extraction from video frames.
- **Long Short-Term Memory (LSTM):** For capturing temporal dependencies across sequences of frames.
- **Softmax Output Layer:** For multi-class classification of actions.

### Architecture Diagram
```mermaid
flowchart LR
    VideoFrames -->|Frame Extraction| CNN --> LSTM -->|Softmax| Output(Predicted Action)

## Technologies Used
- **Python:** Programming language
- **TensorFlow & Keras:** For building and training the deep learning model
- **OpenCV:** For video processing and frame extraction
- **Streamlit:** For deploying the model as a web app
- **NumPy & Pandas:** Data handling
- **Scikit-learn:** One-hot encoding and data preprocessing
## Model Evaluation
The model achieves a high accuracy of 91.16%. Below is a sample classification report:

| Action Class   | Precision | Recall | F1-Score |
| -------------- | --------- | ------ | -------- |
| WalkingWithDog | 0.92      | 0.89   | 0.90     |
| Punch          | 0.91      | 0.93   | 0.92     |
| PushUps        | 0.90      | 0.91   | 0.91     |
| Biking         | 0.94      | 0.92   | 0.93     |

## One-Hot Encoding Example
The one-hot encoding format for classes looks like this:

| Class            | One-Hot Encoding |
| ---------------- | ---------------- |
| 0 (WalkingWithDog) | [1, 0, 0, 0]    |
| 1 (Punch)          | [0, 1, 0, 0]    |
| 2 (PushUps)        | [0, 0, 1, 0]    |
| 3 (Biking)         | [0, 0, 0, 1]    |

## Future Scope
- **Real-Time Detection:** Enhance the model for real-time action detection on live video feeds.
- **Additional Classes:** Include more action classes like Fall Detection or Aggressive Behavior.
- **Alert System Integration:** Implement an automated alert system for real-time notifications.
- **Deployment on Edge Devices:** Optimize the model for edge devices like Raspberry Pi for on-site processing.
