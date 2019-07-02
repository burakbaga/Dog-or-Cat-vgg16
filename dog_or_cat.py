import cv2
import numpy as np
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

vgg16_model = vgg16.VGG16()  # vgg16 modelini getirdik
#Train ,test , valid klasörleri altında Dog ve Cat adında klasörler oluşturup buraya fotoğrafları ayrı ayrı atın
train_path = 'C:\\Users\\bagat\\OneDrive\\Masaüstü\\DCimages\\Train'
valid_path = 'C:\\Users\\bagat\\OneDrive\\Masaüstü\\DCimages\\Valid'
test_path = 'C:\\Users\\bagat\\OneDrive\\Masaüstü\\DCimages\\Test'

# verdiğimiz yoldan batch halinde resimleri alır
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['Cat', 'Dog'],
                                                         batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['Cat', 'Dog'],
                                                        batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['Cat', 'Dog'],
                                                         batch_size=4)

imgs, labels = next(train_batches)

model = Sequential()

# modelin sonra katmanında 1000 tane sınıf var bunun yerine biz 2 sınıf istiyoruz son katmanı silip modei aldık
for layer in vgg16_model.layers[:-1]:
    model.add(layer)  # vgg16 modelini kendi modelimize aktardık

for layer in model.layers:
    layer.trainable = False  # modeli dondurduk update edilmesini istemiyoruz

model.add(Dense(2, activation='softmax'))  # modele katman ekledik

model.summary()

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# batch halinde alınan resimleri veriyoruz
# steps per epoch her epochta kaç adımda verilerin tümünü alıcak toplam veri sayısı bölü batch boyutu
model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5)

test_images, test_labels = next(test_batches)
print(test_batches.class_indices)  # cat 0. indis dog 1. indis

test_labels = test_labels[:, 0]
print(test_labels)

predictions = model.predict_generator(test_batches, steps=1)

for i in predictions:
    print(i)

Categories = ['Cat', 'Dog']


def prepare(im):
    im = cv2.imread(im)
    im = cv2.resize(im, (224, 224))
    return im.reshape(-1, 224, 224, 3)


predictions2 = model.predict(prepare('kopek.jpg'))
print(predictions2)
index = int(np.argmax(predictions2))
print(Categories[index])
