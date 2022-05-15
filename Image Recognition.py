import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# Loading MNIST data

image_height = 28
image_width = 28

color_channels = 1

model_name = "mnist"

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

category_names = list(map(str, range(10)))
# ------------------------------------------------------------

# Process mnist data
print (train_data.shape)

train_data = np.reshape(train_data, (-1, image_height, image_width, color_channels))
print(train_data.shape)

eval_data = np.reshape(eval_data, (-1, image_height, image_width, color_channels))
print(train_data.shape)

# ----------------------------------------------------------------------------------------

# Loading Cifer Data

image_height = 32
image_width = 32

color_channels = 3

model_name = "cifar"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

cifar_path = '/Users/vivekchittepu/PycharmProjects/ImageRecog/cifar-10-data/'

train_data = np.array([])
train_labels = np.array([])

for i in range(1, 4):
    data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
    train_data = np.append(train_data, data_batch[b'data'])
    train_labels = np.append(train_labels, data_batch[b'labels'])

eval_batch = unpickle(cifar_path + 'test_batch')

eval_data = eval_batch[b'data']
eval_labels = eval_batch[b'labels']

category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))
# --------------------------------------------------------------------------------

# Process Cifar data
def process_data(data):
    float_data = np.array(data, dtype=float) / 255.0
    reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))

    transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])
    return transposed_data


train_data = process_data(train_data)
eval_data = process_data(eval_data)


class ConvNet:

    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels],
                                          name="inputs")
        print(self.input_layer.shape)

        conv_layer1 = tf.layers.conv2d(self.input_layer, filters=16, kernel_size=[5, 5], padding="same",
                                       activation=tf.nn.relu)
        print(conv_layer1.shape)
        pooling_layer1 = tf.layers.max_pooling2d(conv_layer1, pool_size=[2, 2], strides=2)
        print(pooling_layer1.shape)

        conv_layer2 = tf.layers.conv2d(pooling_layer1, filters=64, kernel_size=[5, 5], padding="same",
                                       activation=tf.nn.relu)
        print(conv_layer2.shape)
        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)

        flattened_pooling = tf.layers.flatten(pooling_layer_2)
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        print(dense_layer.shape)

        dropout = tf.layers.dropout(dense_layer, rate=0.09, training=True)
        output = tf.layers.dense(dropout, num_classes)
        print(output.shape)

        # Training Model
        self.choice = tf.argmax(output, axis=1)
        self.probability = tf.nn.softmax(output)

        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)

        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=output)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

# -----------------------------------------------------------------------------------------------

# Initialize Variables
training_steps = 5000
batch_size = 32
path = "./" + model_name + "-cnn/"

load_checkpoint = True
performance_graph = np.array([])

tf.reset_default_graph()

features = tf.placeholder(train_data.dtype, train_data.shape)
labels = tf.placeholder(train_labels.dtype, train_labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=train_labels.shape[0])
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

dataset_iterator = dataset.make_initializable_iterator()
next_element = dataset_iterator.get_next()

cnn = ConvNet(image_height, image_width, color_channels, 10)

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)
# ----------------------------------------------------------

# Load Variables

with tf.Session() as sess:
    if load_checkpoint:
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())
    sess.run(dataset_iterator.initializer, feed_dict={features: train_data, labels: train_labels})
    for step in range(training_steps):
        current_batch = sess.run(next_element)

        batch_inputs = current_batch[0]
        batch_labels = current_batch[1]

        sess.run((cnn.train_operation, cnn.accuracy_op),
                 feed_dict={cnn.input_layer: batch_inputs, cnn.labels: batch_labels})

        if step % 10 == 0:
            performance_graph = np.append(performance_graph,
                                          sess.run(cnn.accuracy))

        if step % 1000 == 0 and step > 0:
            current_acc = sess.run(cnn.accuracy)

            print("Accuracy at step " + str(step) + ": " + str(current_acc))

    print("Saving final checkpoint for training session.")
    saver.save(sess, path + model_name, step)
    sess.close()

print("------")
# ------------------------------------------------------------------------------------------------
# Making It Look Good

plt.figure().set_facecolor('white')
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.plot(performance_graph)
plt.show()


with tf.Session() as sess:
    print ("==+")
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    sess.run(tf.local_variables_initializer())
    indexes = np.random.choice(len(eval_data), 10, replace=False)

    rows = 5
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    fig.patch.set_facecolor('white')
    image_count = 0
    for idx in indexes:
        image_count += 1
        sub = plt.subplot(rows, cols, image_count)
        img = eval_data[idx]
        if model_name == "mnist":
            img = img.reshape(28, 28)
        plt.imshow(img)
        guess = sess.run(cnn.choice, feed_dict={cnn.input_layer: [eval_data[idx]]})
        if model_name == "mnist":
            guess_name = str(guess[0])
            actual_name = str(eval_labels[idx])
        else:
            guess_name = category_names[guess[0]]
            actual_name = category_names[eval_labels[idx]]
        print("-------------------------")
        print("Guess: " + guess_name)
        print("Actual: " + actual_name)
        plt.show()
        print("-------------------------")
        sub.set_title("G: " + guess_name + " A: " + actual_name)
    plt.tight_layout()

print("=====")