# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')
    local_zip = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/validation-horse-or-human/')
    zip_ref.close()
    import os
    train_dir = 'tmp/horse-or-human/'
    validation_dir = 'tmp/validation-horse-or-human/'

    train_horses_dir = os.path.join(train_dir, "horses")
    train_humans_dir = os.path.join(train_dir, "humans")
    validation_horses_dir = os.path.join(validation_dir, "horses")
    validation_humans_dir = os.path.join(validation_dir, "humans")

    train_horses_fnames = os.listdir(train_horses_dir)
    train_humans_fnames = os.listdir(train_humans_dir)
    validation_horses_fnames = os.listdir(validation_horses_dir)
    validation_humans_fnames = os.listdir(validation_humans_dir)


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True
)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
    target_size=(300, 300),
    batch_size=10,
    class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
    target_size=(300,300),
    batch_size=10,
    class_mode='binary')

    pretrained_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3),
                               include_top = False)
    pretrained_model.trainable = False
    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        tf.keras.layers.experimental.preprocessing.Resizing(224,224),
        pretrained_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile( tf.keras.optimizers.Adam(lr = 0.001),
              loss='binary_crossentropy',
              metrics=["accuracy"])

    cp_callback = [
               tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_accuracy'),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
]
    with tf.device('/GPU:0'):
      model.fit(train_generator, epochs=20, callbacks= cp_callback, validation_data= validation_generator)
    return model



    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")