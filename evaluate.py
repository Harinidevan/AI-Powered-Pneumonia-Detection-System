from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
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

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")