import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

imagesdata = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels)= imagesdata.load_data()

train_images= tf.keras.utils.normalize(train_images, axis=1)  
test_images = tf.keras.utils.normalize(test_images, axis=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#plotting the first 25 images with plotting size of 10 * 10 inches  
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)   #subplot of 5 * 5 grid where i+1 determines the position of image
    plt.xticks([])  #removes x-axis ticks
    plt.yticks([])  #removes y-axis ticks
    plt.grid(False) #disables the grid
    plt.imshow(train_images[i], cmap=plt.cm.binary)  #displays the ith image from train_images dataset
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = tf.keras.models.Sequential()

#architecture of CNN model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#Train
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

#evalutaion 
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

#visualizing training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

#making predictions
predictions = model.predict(test_images)

# Predicting and displaying random images from the test set
num_images = 5  # Number of random images to display
random_indices = np.random.choice(test_images.shape[0], num_images, replace=False)

for i in random_indices:
    plt.figure(figsize=(2,2))
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[test_labels[i][0]]
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}")
    plt.show()

    print(f"Predicted label for image {i}: {predicted_label}")
    print(f"True label for image {i}: {true_label}")




