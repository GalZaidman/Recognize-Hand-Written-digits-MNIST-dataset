# Gal Zaidman 204311377
# Liran Yehudar 204249411
# Yuri Vainstain ‏ 304610892

import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import math

learning_rate = 0.0001
nb_batches = 13000
mini_batch = 50
img_size  = 28
img_shape = (img_size, img_size)
img_size_flat  = img_size * img_size
nb_of_targets = 10
num_channels = 1
inputs = tf.placeholder(tf.float32, [None, img_size_flat])
x_image = tf.reshape(inputs, [-1, img_size, img_size, num_channels])
targets = tf.placeholder(tf.float32, [None, nb_of_targets])
keep_prob = tf.placeholder(tf.float32)

#heper functions

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    plt.show()

def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {inputs: [image]}
    s = tf.Session()
    s.run(tf.global_variables_initializer())

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = s.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    s.close()
    plt.show()

def write_results_to_file(file_name, model, number_of_weights, cpu_time, recall, precision, f1_score, accuracy):
    with open(file_name, 'a') as results_file:
        results_file.write(
            "{0}"
            "\nnumber of weights = {1}"
            "\nCPU time  :  {2}sec"
            "\nAccuracy : {3} ,"
            "\nPrecision : {4} ,"
            "\nRecall {5} , "
            "\nF-Score : {6}".format(
                model, number_of_weights,  cpu_time, accuracy, precision, recall, precision, f1_score))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def fully_connected_layer(inputs, input_shape, output_shape, activation=tf.nn.relu):
    # definine weights and biases -
    weights = new_weights([input_shape, output_shape])
    biases = new_biases(output_shape)
    # x*W + b <- computation for the layer values
    layer = tf.matmul(inputs, weights) + biases
    if activation != None:
        layer = activation(layer)
    return layer

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,
                   use_relu=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    if(use_relu):
        layer = tf.nn.relu(layer)
    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def runModel(dataset, model, modelname, numOfWeight):
    y_pred = tf.nn.softmax(model)
    #creating the model for run

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=model)
    cost = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #to eval the model

    Pred = tf.argmax(y_pred, 1) # y_pred_cls
    Desired = tf.argmax(targets, 1)
    y_true = dataset.test.cls
    correct_prediction = tf.equal(Pred, Desired)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #session start

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start_time = time.time()

    #training the model
    for i in range(nb_batches):
        batch = dataset.train.next_batch(mini_batch)  # get next mini batch
        sess.run(train_step, feed_dict={inputs: batch[0], targets: batch[1], keep_prob: 0.5})

    end_time = time.time()
    time_dif = end_time - start_time
    cpu_time = str(timedelta(seconds=int(round(time_dif))))
    test_Accuracy, test_pred = sess.run([accuracy, Pred], feed_dict={inputs: dataset.test.images, targets: dataset.test.labels, keep_prob: 1.0})
    sess.close()
    # session start
    acc = sk.accuracy_score(y_true, test_pred)
    precision = sk.precision_score(y_true, test_pred, average='macro')
    recall = sk.recall_score(y_true, test_pred, average='macro')
    f1_score = sk.f1_score(y_true, test_pred, average='macro')

    print("Results for model {} : ".format(modelname))
    print("Time it took to train : " + cpu_time)
    print("Recall   : {} , Precision   : {} , F_Score   : {} , Accuracy   : {}  ".format(recall, precision, f1_score, test_Accuracy))
    write_results_to_file('results for {}.txt'.format(modelname),modelname, numOfWeight, cpu_time, recall, precision, f1_score, test_Accuracy)

#   A:
def logistic_regresion():
    # creatr a logistic regresion z = W*x + b
    W = new_weights([img_size_flat, nb_of_targets])
    b = new_biases(nb_of_targets)
    z = tf.matmul(inputs, W) + b  # a single layer
    total_weights = img_size_flat * nb_of_targets + nb_of_targets
    return (z,total_weights)

#   B:
def Two_Fully_Connected_Layers_200_ReLu():
    nb_n_in_hlayer = 200
    layer_fc1 = fully_connected_layer(inputs=inputs,
                                      input_shape=img_size_flat,
                                      output_shape=nb_n_in_hlayer)
    layer_fc2 = fully_connected_layer(inputs=layer_fc1,
                                      input_shape=nb_n_in_hlayer,
                                      output_shape=nb_n_in_hlayer)
    layer_fc3 = fully_connected_layer(inputs=layer_fc1,
                                      input_shape=nb_n_in_hlayer,
                                      output_shape=nb_of_targets,
                                      activation=None)
    total_weights = img_size_flat * nb_n_in_hlayer + nb_n_in_hlayer + nb_n_in_hlayer * nb_n_in_hlayer + nb_n_in_hlayer \
                    + nb_n_in_hlayer * nb_of_targets + nb_of_targets
    return layer_fc3,total_weights

#C
def CNN1():
    nb_n_in_hlayer = 1024
    num_filters1 = 32
    filter_size1 = 5
    layer_conv1 = new_conv_layer(input=x_image,
                                                num_input_channels=num_channels,
                                                filter_size=filter_size1,
                                                num_filters=num_filters1,
                                                use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv1)

    layer_fc1 = fully_connected_layer(inputs=layer_flat,
                                      input_shape=num_features,
                                      output_shape=nb_n_in_hlayer)

    layer_fc2 = fully_connected_layer(inputs=layer_fc1,
                                      input_shape=nb_n_in_hlayer,
                                      output_shape=nb_of_targets,
                                      activation=None)
    num_of_w_in_c1 = (filter_size1*filter_size1*1 +1)*num_filters1
    num_of_w_in_fc1 =  num_features*nb_n_in_hlayer + nb_n_in_hlayer
    num_of_w_in_fc2 =  nb_n_in_hlayer*nb_of_targets + nb_of_targets
    total_weights = num_of_w_in_c1 + num_of_w_in_fc1 + num_of_w_in_fc2

    return layer_fc2,total_weights

#D + E
def CNN2(dropOut=False, needToPlot= False):
    nb_n_in_hlayer = 1024
    num_filters1 = 32
    filter_size1 = 5
    num_filters2 = 64
    filter_size2 = 5

    layer_conv1 = new_conv_layer(input= x_image,
                                                num_input_channels=num_channels,
                                                filter_size=filter_size1,
                                                num_filters=num_filters1,
                                                use_pooling=True)

    layer_conv2 = new_conv_layer(input=layer_conv1,
                                                num_input_channels=num_filters1,
                                                filter_size=filter_size2,
                                                num_filters=num_filters2,
                                                use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)

    layer_fc1 = fully_connected_layer(inputs=layer_flat,
                                      input_shape=num_features,
                                      output_shape=nb_n_in_hlayer)

    if(dropOut):
        layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob)

    layer_fc2 = fully_connected_layer(inputs=layer_fc1,
                                      input_shape=nb_n_in_hlayer,
                                      output_shape=nb_of_targets,
                                      activation=None)

    num_of_w_in_c1 = (filter_size1*filter_size1*1 +1)*num_filters1
    num_of_w_in_c2 =  (filter_size1*filter_size1*num_filters1 +1)*num_filters2
    num_of_w_in_fc1 =  num_features*nb_n_in_hlayer + nb_n_in_hlayer
    num_of_w_in_fc2 =  nb_n_in_hlayer*nb_of_targets + nb_of_targets
    total_weights = num_of_w_in_c1 + num_of_w_in_c2 + num_of_w_in_fc1 + num_of_w_in_fc2
    if(needToPlot):
        layer_conv1_n_relu = new_conv_layer(input= x_image,
                                                            num_input_channels=num_channels,
                                                            filter_size=filter_size1,
                                                            num_filters=num_filters1,
                                                            use_pooling=False,
                                                            use_relu=False)
        layer_conv2_no_relu = new_conv_layer(input=layer_conv1,
                                                            num_input_channels=num_filters1,
                                                            filter_size=filter_size2,
                                                            num_filters=num_filters2,
                                                            use_pooling=False,
                                                            use_relu=False)
        return (layer_fc2,total_weights,  layer_conv1_n_relu, layer_conv1, layer_conv2_no_relu, layer_conv2)
    return layer_fc2, total_weights

if __name__ == "__main__":
    import MNIST as dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist.test.cls = np.array([label.argmax() for label in mnist.test.labels])

    A,numOfWeights = logistic_regresion()
    runModel(mnist, A, "Logistic Regresion", numOfWeights)
    B,numOfWeights = Two_Fully_Connected_Layers_200_ReLu()
    runModel(mnist, B, "Two Fully Connected Layers", numOfWeights)
    C =CNN1()
    runModel(mnist, C[0], "CNN with one convolution layer",C[1])
    D =CNN2()
    runModel(mnist, D[0], "CNN with two convolution layers",D[1])
    E = CNN2(dropOut=True, needToPlot= True)
    runModel(mnist,E[0], "CNN with two convolution layers and dropout",E[1])
    mini_batch = 100
    runModel(mnist, E[0], "CNN with two convolution layers and dropout 100 mini-batch",E[1])
    a = {}
    a.get()
    # REMOVE # TO PLOT THE IMAGE
    #image = mnist.test.images[1]
    #plot_image(image)
    #plot_conv_layer(layer=E[2], image=image)
    #plot_conv_layer(layer=E[3], image=image)
    #plot_conv_layer(layer=E[4], image=image)
    #plot_conv_layer(layer=E[5], image=image)


