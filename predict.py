import cv2
import tensorflow as tf
import os
import numpy as np
from basic_model import model

def prediction(image):
    #model_1 = model()
    model_folder = 'checkpoints'
    #image = 'test_sample.png'
    img = cv2.imread(image)
    session = tf.Session()
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 3)
    labels = np.zeros((1, 4))

    # create an object to save the graph
    saver = tf.train.import_meta_graph(os.path.join(model_folder, 'model.meta'))

    # For restore the model
    saver.restore(session, tf.train.latest_checkpoint('./checkpoints'))

    # Create graph object for getting the same network architecture
    graph = tf.get_default_graph()

    # Get the last layer of the network by it's name which includes all the previous layers too
    network = graph.get_tensor_by_name("add_4:0")

    # create placeholders to pass the image and get output labels
    im_ph= graph.get_tensor_by_name("Placeholder:0")
    label_ph = graph.get_tensor_by_name("Placeholder_1:0")

    # Inorder to make the output to be either 0 or 1.
    network=tf.nn.sigmoid(network)

    # Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {im_ph: img, label_ph: labels}
    result=session.run(network, feed_dict=feed_dict_testing)
    return result


if __name__=="__main__":
    print(prediction('test_image.jpg'))



