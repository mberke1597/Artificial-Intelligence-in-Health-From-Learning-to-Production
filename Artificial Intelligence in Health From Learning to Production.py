import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Veri yolları
aca_dir = '/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/colon_image_sets/colon_aca'
n_dir = '/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/colon_image_sets/colon_n'

# İlk 200 resmi alabilmek için dosya adlarını al
aca_files = os.listdir(aca_dir)[:200]
n_files = os.listdir(n_dir)[:200]

# Veri artırma (augmentation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

# İlk 200 veriyi kullanarak veri setleri oluşturma
def create_dataframe(base_dir, files, label):
    data = []
    for file in files:
        data.append([os.path.join(base_dir, file), str(label)])  # Label values as strings
    return pd.DataFrame(data, columns=['filepath', 'label'])

aca_data = create_dataframe(aca_dir, aca_files, 1)
n_data = create_dataframe(n_dir, n_files, 0)

data = pd.concat([aca_data, n_data]).reset_index(drop=True)

# Verileri karıştır ve train/test olarak ayır
train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Model oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Modelin derlenmesi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin eğitilmesi
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Eğitim ve doğrulama doğruluğunun görselleştirilmesi
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Eğitim Doğruluğu')
plt.plot(epochs, val_acc, 'b', label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Eğitim Kaybı')
plt.plot(epochs, val_loss, 'b', label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()
plt.show()


