import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Force TensorFlow to use the CPU
tf.config.set_visible_devices([], 'GPU')

# Checking TensorFlow version
print(tf.__version__)

# Data preprocessing (training & evaluation set)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
    '../datasets/catsdogs/training_set',
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    '../datasets/catsdogs/test_set',
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary')

# Initializing the CNN
cnn = tf.keras.models.Sequential()

# Convolution and Pooling Layers
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[32, 32, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the CNN with adjusted learning rate and gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, clipvalue=1.0)
cnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Function to make predictions and check if all are correct
def make_predictions_and_check_accuracy(image_paths):
    all_correct = True
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(32, 32))
        img_array = image.img_to_array(img)

        # Normalize the image (scale pixel values to the range [0, 1])
        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        result = cnn.predict(img_array)

        # Determine the category and calculate confidence
        if result[0][0] >= 0.5:
            predicted_category = 'dog'
            confidence = result[0][0] * 100
        else:
            predicted_category = 'cat'
            confidence = (1 - result[0][0]) * 100

        # Check if the prediction is correct by comparing it to the filename
        actual_category = 'dog' if 'dog' in os.path.basename(image_path).lower() else 'cat'
        if predicted_category != actual_category:
            all_correct = False

        # Print the result for each image with confidence
        print(f'Prediction for {os.path.basename(image_path)} is "{predicted_category}" with confidence: {confidence:.2f}% (Actual: "{actual_category}")')

    return all_correct

# Custom callback to make predictions after each epoch and count correct epochs
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.correct_epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEnd of epoch {epoch + 1}")
        all_correct = make_predictions_and_check_accuracy(self.image_paths)
        if all_correct:
            self.correct_epoch_count += 1
            print(f"All predictions were correct for this epoch!")
        else:
            print(f"Not all predictions were correct for this epoch.")

        print(f'Correct epochs so far: {self.correct_epoch_count}/{epoch + 1}\n')

# Paths to the images you want to predict after each epoch
image_paths = [
    '../datasets/catsdogs/single_prediction/dog.jpg',
    '../datasets/catsdogs/single_prediction/cat.jpg'
]

# Create an instance of the callback
prediction_callback = PredictionCallback(image_paths)

# Training the CNN and capturing history
history = cnn.fit(
    x=train_set,
    validation_data=test_set,
    epochs=30,
    callbacks=[prediction_callback]
)

# Plotting the results
# Training and validation accuracy over epochs
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('cnn_cd_e30_accuracy.png')

# Training and validation loss over epochs
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('cnn_cd_e30_loss.png')
