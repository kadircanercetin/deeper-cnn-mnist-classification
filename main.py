from numpy import std, mean
import numpy as np
from matplotlib import pyplot
import cv2
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from googlenet import GoogleNet
from resnet34 import Resnet34
from vgg16 import Vgg16Cifar10


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainX = prep_pixels(trainX)
    testX = prep_pixels(testX)
    # one hot encode target values
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    # split training data for validation
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=1)

    return trainX, trainY, valX, valY, testX, testY


# scale pixels
def prep_pixels(val):
    # convert from integers to floats
    val_norm = val.astype('float32')
    # normalize to range 0-1
    val_norm = val_norm / 255.0
    # return normalized images
    return val_norm


def plot_diagnostics(history, file_name):
    # summarize history for accuracy
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.savefig('diagnostics/' + file_name + '_acc.png')
    pyplot.show()
    # summarize history for loss
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.savefig('diagnostics/' + file_name + '_loss.png')
    pyplot.show()


# summarize model performance
def summarize_performance(scores, model_name):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.savefig('performances/' + model_name + '.png')
    pyplot.show()


# convert 28x28 grayscale to 48x48 rgb channels
def to_rgb(img, dim):
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    return img_rgb


def resize_and_convert_to_rgb(dataset, weight, height):
    dim = (weight, height)
    rgb_list = []
    for i in range(len(dataset)):
        rgb = to_rgb(dataset[i], dim)
        rgb_list.append(rgb)
    rgb_arr = np.stack([rgb_list], axis=4)
    return np.squeeze(rgb_arr, axis=4)


def main():
    #  VGG16
    model = Vgg16Cifar10(imagenet_pre_trained=True)
    history = model.train()
    # plot the training loss and accuracy
    plot_diagnostics(history, model.name)
    # get the test data already loaded on initialization
    x_test = model.testX
    y_test = model.testY
    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x, 1) != np.argmax(y_test, 1)
    loss = sum(residuals) / len(residuals)
    print("the validation 0/1 loss is: ", loss)


    # load MNIST dataset for googlenet and resnet34
    trainX, trainY, valX, valY, testX, testY = load_dataset()

    #  GoogleNet
    img_dim = (28, 28, 1)
    model = GoogleNet.build_model(img_dim, 10)
    history = model.fit(trainX, trainY, epochs=10, validation_data=(valX, valY), batch_size=32, verbose=2)
    _, acc = model.evaluate(testX, testY, verbose=2)
    print('Test accuracy: %.3f' % (acc * 100.0))
    # plot the training loss and accuracy
    plot_diagnostics(history, model.name)
    # make predictions on the test set
    preds = model.predict(testX)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1)))

    #  Resnet34
    model = Resnet34.build_model(img_dim, 10)
    history = model.fit(trainX, trainY, epochs=10, validation_data=(valX, valY), batch_size=128, verbose=2)
    loss, acc = model.evaluate(testX, testY, verbose=2)
    print('Test accuracy: %.3f' % (acc * 100.0))
    # plot the training loss and accuracynip
    plot_diagnostics(history, model.name)
    # make predictions on the test set
    preds = model.predict(testX)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1)))







if __name__ == "__main__":
    main()