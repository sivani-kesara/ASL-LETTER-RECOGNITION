from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
train_dir = 'processed_images/train'
test_dir = 'processed_images/test'
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=26, activation='softmax')) 
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(sz, sz),
                                                 batch_size=64,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(sz , sz),
                                            batch_size=64,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
print("Model training started...")
classifier.fit(
        training_set,
        steps_per_epoch=training_set.n//training_set.batch_size,
        epochs=10,
        validation_data=test_set,
        validation_steps=test_set.n//test_set.batch_size)
classifier.save('mymodel.h5')
print('Model Saved')