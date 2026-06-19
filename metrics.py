from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import numpy as np

# Load model
model = load_model("densenet_pneumonia.h5")

# Load data
datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    "small_dataset/train",
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Predictions
pred_probs = model.predict(test_data)
preds = (pred_probs > 0.5).astype(int)

# True labels
y_true = test_data.classes

# Metrics
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
auc = roc_auc_score(y_true, pred_probs)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {auc:.4f}")