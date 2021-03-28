import matplotlib.pyplot as plt
import seaborn as sns

import keras_efficientnets
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      )

test_datagen = ImageDataGenerator(rescale=1./255)

target_size = (224, 224, 3)
batch_size = 128
train_generator = train_datagen.flow_from_directory(
        "train", 
        target_size=target_size[:2],  
        batch_size=batch_size,
        class_mode='categorical',
      color_mode="rgb")

validation_generator = test_datagen.flow_from_directory(
        "test",
        target_size=target_size[:2],
        batch_size=batch_size,
        class_mode='categorical',
      color_mode="rgb")

effnet = keras_efficientnets.EfficientNetB0(
    input_shape=(224,224,3),
    weights='imagenet',
    include_top=False
)
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential
from keras.layers import Activation, Dense, LeakyReLU, ReLU
from keras.optimizers import Adam

def build_model():
    model = Sequential()
    # model.add(MaxPool2D(2))
    model.add(effnet)
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(10, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(18, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0001,decay=1e-6),
        metrics=['accuracy']
    )
    
    return model

model = build_model()
model.summary()

results=model.fit(
    train_generator,
    epochs=50,
    validation_data = validation_generator,
    batch_size=batch_size,
    callbacks=[early_stopper],
    verbose=1
)

# graph training and validation loss
losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()
plt.show()
print("Ending Loss:", losses['val_loss'][len(losses['val_loss'])-1])
# graph training and validation accuracy
losses[['accuracy', 'val_accuracy']].plot()
print("Ending Accuracy:", 100*losses['val_accuracy'][len(losses['val_accuracy'])-1])
plt.show()

model.save("model.h5")

model.evaluate(validation_generator)

# reformat the labels, potentially using something like np.argmax
predictions_raw = model.predict(validation_generator)
predictions = [np.argmax(i) for i in predictions_raw]

#Convert numbers to class names
class_names = list(validation_generator.class_indices.keys())
from tqdm import tqdm
ys = []
predictions=[]
for i in tqdm(range(len(validation_generator))):
  x, y = next(validation_generator)
  predictions.extend(model(x))
  ys.extend(y)
ys=np.array(ys)
predictions=[class_names[np.argmax(i)] for i in predictions]
y_test_names = [class_names[np.argmax(i)] for i in ys]
print(y_test_names)

print(len(predictions))
len(y_test_names)

# print classification report and confusion matrix (normalized and not normalized)
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(y_test_names, predictions)
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test_names, predictions, labels=list(validation_generator.class_indices.keys()))
conf_matrix_normalized = confusion_matrix(y_test_names, predictions, normalize='true', labels=list(validation_generator.class_indices.keys()))
print(conf_matrix)
print()
print("Confusion Matrix Normalized:")
print(conf_matrix_normalized)

plt.figure(figsize=(15,15))
sns.heatmap(conf_matrix_normalized, annot=True)
plt.show()

