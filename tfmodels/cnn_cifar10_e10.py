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
train_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(
    '../datasets/cifar10/training_set',
    target_size=(32, 32),
    batch_size=80,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    '../datasets/cifar10/test_set',
    target_size=(32, 32),
    batch_size=80,
    class_mode='categorical')

# Get class indices (mapping from class names to class indices)
class_indices = train_set.class_indices
class_indices_rev = {v: k for k, v in class_indices.items()}  # Reverse mapping for predictions

# Initializing the CNN
cnn = tf.keras.models.Sequential()

# Convolutional Layer Block 1
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[32, 32, 3]))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Convolutional Layer Block 2
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Convolutional Layer Block 3
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Layers
cnn.add(tf.keras.layers.Dense(units=256))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))

cnn.add(tf.keras.layers.Dense(units=128))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Activation('relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compile the CNN with adjusted learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)
cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Function to make predictions and check if all are correct
def make_predictions_and_check_accuracy(image_folder, class_indices_rev):
    all_correct = True
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(32, 32))
        img_array = image.img_to_array(img)

        # Normalize the image (scale pixel values to the range [0, 1])
        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        result = cnn.predict(img_array)

        # Determine the predicted category
        predicted_category_idx = np.argmax(result, axis=1)[0]
        predicted_category = class_indices_rev[predicted_category_idx]
        confidence = result[0][predicted_category_idx] * 100

        # Determine the actual category from the image filename
        actual_category = next((cat for cat in class_indices_rev.values() if cat in os.path.basename(image_path)), None)

        if actual_category is None or predicted_category != actual_category:
            all_correct = False

        # Print the result for each image with confidence
        print(f'Prediction for {os.path.basename(image_path)} is "{predicted_category}" with confidence: {confidence:.2f}% (Actual: "{actual_category}")')

    return all_correct

# Custom callback to make predictions after each epoch and count correct epochs
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_folder, class_indices_rev):
        self.image_folder = image_folder
        self.class_indices_rev = class_indices_rev
        self.correct_epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEnd of epoch {epoch + 1}")
        all_correct = make_predictions_and_check_accuracy(self.image_folder, self.class_indices_rev)
        if all_correct:
            self.correct_epoch_count += 1
            print(f"All predictions were correct for this epoch!")
        else:
            print(f"Not all predictions were correct for this epoch.")

        print(f'Correct epochs so far: {self.correct_epoch_count}/{epoch + 1}\n')

# Set the folder path to the images you want to predict after each epoch
image_folder = '../datasets/cifar10/single_prediction/'

# Create an instance of the callback
prediction_callback = PredictionCallback(image_folder, class_indices_rev)

# Training the CNN and capturing history
history = cnn.fit(
    x=train_set,
    validation_data=test_set,
    epochs=10,
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
plt.savefig('cnn_cifar10_e10_accuracy.png')

# Training and validation loss over epochs
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('cnn_cifar10_e10_loss.png')
