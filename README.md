# Human Action Detection Using Deep Learning 📹
## Overview
This project aims to develop a Human Action Detection system using deep learning techniques. The model can detect violent actions such as Punch, PushUps, and everyday activities like WalkingWithDog and Biking from video sequences. It uses Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) for temporal sequence learning.

## Project Goal
To create a reliable alert system that can automatically identify potentially harmful actions in real-time and notify authorities, aiming to ensure timely intervention and improve safety.

## Key Features
- Detects and classifies human actions from videos.
- Achieves an accuracy of 91.16% on the validation dataset.
- Real-time action detection using video feed or pre-recorded videos.
- Deployed using Streamlit for user-friendly interaction.

## Dataset
The dataset includes various human action videos categorized into four main classes:
- WalkingWithDog
- Punch
- PushUps
- Biking

## Data Preprocessing
The video frames are resized, normalized, and fed into a CNN + LSTM model for processing. The labels are one-hot encoded for multi-class classification.

## Model Architecture
The model combines:
- **Convolutional Neural Network (CNN):** For spatial feature extraction from video frames.
- **Long Short-Term Memory (LSTM):** For capturing temporal dependencies across sequences of frames.
- **Softmax Output Layer:** For multi-class classification of actions.

### Model Architecture Description
1. **Video Frames** are extracted from video sequences.
2. The frames pass through the **CNN** for feature extraction.
3. The extracted features are processed by the **LSTM** to learn temporal dependencies.
4. The output from LSTM is passed through a **Softmax** layer to classify the detected action.

## Technologies Used
- **Python**: Programming language
- **TensorFlow & Keras**: For building and training the deep learning model
- **OpenCV**: For video processing and frame extraction
- **Streamlit**: For deploying the model as a web app
- **NumPy & Pandas**: For data handling
- **Scikit-learn**: For one-hot encoding and data preprocessing

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
