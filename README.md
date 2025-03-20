# Emotion-Recognition-ML
🚀 README for Emotion Recognition Assignment
This repository contains a machine learning project on multimodal emotion recognition using video and text data. The project was developed as part of a Machine Learning Hackathon and explores feature extraction from video frames (ResNet50) and textual utterances (BERT) for emotion classification.

📌 Overview
Video Processing: Extracts deep visual features from video clips using a pre-trained ResNet50 model.
Text Processing: Uses BERT embeddings to extract semantic meaning from speech transcripts.
Feature Fusion: Implements early fusion by concatenating visual and textual features.
Model Selection: Trains an SVM classifier with a linear kernel on the fused features.
Evaluation: Assesses performance using accuracy, precision, recall, and F1-score.

📂 Dataset
The dataset consists of:
Train Data: train_emotion.csv with text dialogues and corresponding video clips.
Test Data: test_emotion.csv, similar to train data but without emotion labels.

Key columns:
Emotion: Target label (e.g., Neutral, Happy, Angry).
Utterance: Text spoken in the video clip.
video_clip_path: Path to the associated video file.

🔍 Approach

1️⃣ Feature Extraction:
Video: Extracted 2048-dimensional feature vectors using ResNet50 on video frames.
Text: Processed 768-dimensional BERT embeddings for textual utterances.

2️⃣ Feature Fusion:
Used early fusion by concatenating video and text features into a 2800-dimensional vector.

3️⃣ Model Training:
Implemented Support Vector Machine (SVM) and Logistic Regression classifiers.
Normalized features using StandardScaler before training.

4️⃣ Evaluation Metrics:
Accuracy, Precision, Recall, F1-score
Confusion Matrix & Classification Report

📊 Results
SVM Model Performance: Achieved 61% accuracy.
Test Set Predictions: Stored in a CSV file for evaluation.
