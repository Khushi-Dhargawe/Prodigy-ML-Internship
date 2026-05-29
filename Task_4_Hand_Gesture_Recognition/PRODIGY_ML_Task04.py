# ============================================================
# PRODIGY INFOTECH — MACHINE LEARNING INTERNSHIP
# Task 04: Hand Gesture Recognition using CNN
# Dataset: https://www.kaggle.com/gti-upm/leapgestrecog
# Tools: Python, TensorFlow/Keras, OpenCV, Matplotlib
# Run in: Google Colab (GPU recommended)
# ============================================================

import os, random, zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Colab: Mount Drive & Download Dataset 
# Uncomment and run these lines in Google Colab:
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p ~/.kaggle
# !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d gti-upm/leapgestrecog
# !unzip leapgestrecog.zip -d Data_hand_gesture

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("=" * 60)
print("TASK 04 — HAND GESTURE RECOGNITION (CNN)")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# 1. Dataset Path 
DATASET_PATH = 'Data_hand_gesture'  # Update if different
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

# 2. Data Generators with Augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"\nClasses ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Training samples  : {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# 3. Sample Images 
imgs, lbls = next(train_generator)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i])
    ax.set_title(CLASS_NAMES[np.argmax(lbls[i])], fontsize=9)
    ax.axis('off')
plt.suptitle('Sample Training Images', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_sample_images.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig1_sample_images.png")

# 4. CNN Architecture
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    # Classifier Head
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. Callbacks 
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# 6. Train 
print("\nTraining CNN...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 7. Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue', lw=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='coral', lw=2)
axes[0].set_title('Model Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', color='steelblue', lw=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', color='coral', lw=2)
axes[1].set_title('Model Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('CNN Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_training_curves.png")

# 8. Evaluation 
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\n[ FINAL RESULTS ]\nValidation Accuracy: {val_acc*100:.2f}%\nValidation Loss    : {val_loss:.4f}")

# 9. Confusion Matrix 
val_generator.reset()
y_pred_prob = model.predict(val_generator, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_generator.classes[:len(y_pred)]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix\nVal Accuracy: {val_acc*100:.2f}%', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_confusion_matrix.png")

print(f"\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 10. Save Model 
model.save('gesture_recognition_model.h5')
print("\nModel saved: gesture_recognition_model.h5")
print("\n[TASK 04 COMPLETE]")
