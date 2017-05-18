import tensorflow as tf
import numpy as np
import sys
import os
import pickle
import time
import getopt
import cv2

from alexnet import AlexNet
from caffe_classes import class_names
from datagenerator import ImageDataGenerator



class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

def initialize_variables(exist, file_name):
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    keys = ['cov1','cov2','fc1','fc2','fc3']
    if (exist == 1):
        with open(file_name, 'rb') as f:
            (weights_val, biases_val) = pickle.load(f)
        weights = {
            'cov1': tf.Variable(weights_val['cov1']),
            'cov2': tf.Variable(weights_val['cov2']),
            'fc1': tf.Variable(weights_val['fc1']),
            'fc2': tf.Variable(weights_val['fc2']),
            'fc3': tf.Variable(weights_val['fc3'])
        }
        biases = {
            'cov1': tf.Variable(biases_val['cov1']),
            'cov2': tf.Variable(biases_val['cov2']),
            'fc1': tf.Variable(biases_val['fc1']),
            'fc2': tf.Variable(biases_val['fc2']),
            'fc3': tf.Variable(biases_val['fc3'])
        }
    else:
        weights = {
            'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 64],
                                                        stddev=5e-2)),
            'cov2': tf.Variable(tf.truncated_normal([5, 5, 64, 64],
                                                        stddev=5e-2)),
            'fc1': tf.Variable(tf.truncated_normal([6 * 6 * 64, 384],
                                                        stddev=0.04)),
            'fc2': tf.Variable(tf.random_normal([384, 192],
                                                        stddev=0.04)),
            'fc3': tf.Variable(tf.random_normal([192, NUM_CLASSES],
                                                        stddev=1/192.0))
        }
        biases = {
            'cov1': tf.Variable(tf.constant(0.1, shape=[64])),
            'cov2': tf.Variable(tf.constant(0.1, shape=[64])),
            'fc1': tf.Variable(tf.constant(0.1, shape=[384])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[192])),
            'fc3': tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]))
        }
    return (weights, biases)

def prune_weights(cRates, weights, weights_mask, biases, biases_mask, mask_dir, f_name):
    keys = ['cov1','cov2','fc1','fc2','fc3']
    new_mask = {}
    for key in keys:
        w_eval = weights[key].eval()
        threshold_off = 0.9*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        threshold_on = 1.1*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        mask_off = np.abs(w_eval) < threshold_off
        mask_on = np.abs(w_eval) > threshold_on
        new_mask[key] = np.logical_or(((1 - mask_off) * weights_mask[key]),mask_on).astype(int)
    with open(mask_dir + f_name, 'wb') as f:
        pickle.dump((new_mask,biases_mask), f)

def initialize_weights_mask(first_time_training, mask_dir, file_name):
    NUM_CHANNELS = 3
    NUM_CLASSES = 10
    if (first_time_training == 1):
        print('setting initial mask value')
        weights_mask = {
            'cov1': np.ones([11, 11, NUM_CHANNELS, 96]),
            'cov2': np.ones([5, 5, 96, 256]),
            'cov3': np.ones([3, 3, 256, 384]),
            'cov4': np.ones([3, 3, 384, 384]),
            'cov5': np.ones([3, 3, 384, 256]),
            'fc6': np.ones([6 * 6 * 256, 4096]),
            'fc7': np.ones([4096, 4096]),
            'fc8': np.ones([4096, NUM_CLASSES])
        }
        biases_mask = {
            'cov1': np.ones([64]),
            'cov2': np.ones([64]),
            'cov3': np.ones([64]),
            'cov4': np.ones([64]),
            'cov5': np.ones([64]),
            'fc6': np.ones([384]),
            'fc7': np.ones([192]),
            'fc8': np.ones([NUM_CLASSES])
        }

        # with open(mask_dir + 'maskcov0cov0fc0fc0fc0.pkl', 'wb') as f:
        #     pickle.dump((weights_mask,biases_mask), f)
    else:
        with open(mask_dir + file_name,'rb') as f:
            (weights_mask, biases_mask) = pickle.load(f)
    print('weights set')
    return (weights_mask, biases_mask)

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'].eval())
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'])
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1', 'fc2','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

def save_pkl_model(weights, biases, save_dir ,f_name):
    # name = os.path.join(data_path, "cifar-10-batches-py/", filename)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    keys = ['cov1','cov2','fc1','fc2','fc3']
    weights_val = {}
    biases_val = {}
    for key in keys:
        weights_val[key] = weights[key].eval()
        biases_val[key] = biases[key].eval()
    with open(save_dir + f_name, 'wb') as f:
        print('Created a pickle file')
        pickle.dump((weights_val, biases_val), f)

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def compute_file_name(thresholds):
    keys_cov = ['cov1', 'cov2']
    keys_fc = ['fc1', 'fc2', 'fc3']
    name = ''
    for key in keys_cov:
        name += 'cov'+ str(int(thresholds[key]*10))
    for key in keys_fc:
        if (key == 'fc1'):
            name += 'fc'+ str(int(thresholds[key]*100))
        else:
            name += 'fc'+ str(int(thresholds[key]*10))
    return name


def main(argv = None):
    if (argv is None):
        argv = sys.argv
    try:
        try:
            opts = argv
            first_time_load = True
            parent_dir = './'
            keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
            prune_thresholds = {}
            WITH_BIASES = False
            save_for_next_iter = False
            TEST = False
            for key in keys:
                prune_thresholds[key] = 0.

            for item in opts:
                print (item)
                opt = item[0]
                val = item[1]
                if (opt == '-cRates'):
                    cRates = val
                if (opt == '-first_time'):
                    first_time_load = val
                if (opt == '-file_name'):
                    file_name = val
                if (opt == '-train'):
                    TRAIN = val
                if (opt == '-prune'):
                    PRUNE = val
                if (opt == '-test'):
                    TEST = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-lr'):
                    lr = val
                if (opt == '-with_biases'):
                    WITH_BIASES = val
                if (opt == '-lambda1'):
                    lambda_1 = val
                if (opt == '-lambda2'):
                    lambda_2 = val
                if (opt == '-save'):
                    save_for_next_iter = val
                if (opt == '-org_file_name'):
                    org_file_name = val


            print('pruning thresholds are {}'.format(prune_thresholds))
        except getopt.error, msg:
            raise Usage(msg)
        epochs = 100
        dropout = 0.5
        batch_size = 1
        num_classes = 1000

        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        INITIAL_LEARNING_RATE = 0.001
        INITIAL_LEARNING_RATE = lr
        LEARNING_RATE_DECAY_FACTOR = 0.1
        NUM_EPOCHS_PER_DECAY = 350.0
        DISPLAY_FREQ = 50
        TEST = 1
        TRAIN_OR_TEST = 0
        NUM_CHANNELS = 3
        first_time_load = 1

        mask_dir = parent_dir
        weights_dir = parent_dir
        LOCAL_TEST = 1


        file_name_part = compute_file_name(cRates)

        if (save_for_next_iter):
            (weights_mask, biases_mask)= initialize_weights_mask(first_time_load, mask_dir, 'mask'+org_file_name + '.pkl')
        else:
            (weights_mask, biases_mask)= initialize_weights_mask(first_time_load, mask_dir, 'mask'+file_name_part + '.pkl')


        meta_data_dir = '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC'
        index_file_dir = '/local/scratch/share/ImageNet/ILSVRC/ImageSets/CLS-LOC/'
        index_file_dir = 'cpu_test_data/'
        if (TRAIN):
            train_file_txt = index_file_dir + 'train.txt'
            val_file_txt = index_file_dir + 'val.txt'
            test_file_txt = index_file_dir + 'test.txt'
        else:
            test_file_txt = index_file_dir + 'test.txt'

        # if (first_time_load):
        #     PREV_MODEL_EXIST = 0
        #     weights, biases = initialize_variables(PREV_MODEL_EXIST, '')
        # else:
        #     PREV_MODEL_EXIST = 1
        #     if (save_for_next_iter):
        #         file_name_part = org_file_name
        #     else:
        #         file_name_part = compute_file_name(cRates)
        #     weights, biases = initialize_variables( PREV_MODEL_EXIST,
        #                                             weights_dir + 'weights' + file_name_part + '.pkl')
        #

        x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        y = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)

        # initilize the model from the class constructer
        model = AlexNet(x, keep_prob, num_classes)

        score = model.fc8
        softmax = tf.nn.softmax(score)

        with tf.name_scope("cross_ent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

        with tf.name_scope("train"):
            # l1_norm = lambda_1 * l1
            # l2_norm = lambda_2 * l2
            # regulization_loss = l1_norm + l2_norm

            opt = tf.train.AdamOptimizer(lr)
            # loss_value = tf.reduce_mean(cross_entropy) + regulization_loss
            grads = opt.compute_gradients(loss)
            org_grads = [(ClipIfNotNone(grad), var) for grad, var in grads]
            train_step = opt.apply_gradients(org_grads)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
        #     weights_new[key] = weights[key] * tf.constant(weights_mask[key], dtype=tf.float32)

        #
        # l1, l2, pred = cov_network(images, weights_new, biases, keep_prob)
        # _, _, test_pred = cov_network(test_images, weights_new, biases, keep_prob)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y)
        #

        # new_grads = mask_gradients(weights, org_grads, weights_mask, biases, biases_mask, WITH_BIASES)

        # Apply gradients.


        init = tf.global_variables_initializer()
        accuracy_list = np.zeros(20)
        train_acc_list = []

        # Launch the graph
        print('Graph launching ..')

        if (TRAIN):
            train_generator = ImageDataGenerator(train_file_txt,
                                                 horizontal_flip = True, shuffle = True)
            val_generator = ImageDataGenerator(val_file_txt, shuffle = False)

        # test_generator = ImageDataGenerator(test_file_txt, shuffle = False)

        if (TRAIN):
            # Get the number of training/validation steps per epoch
            train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
            val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

        # test_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int16)

        with tf.Session() as sess:
            sess.run(init)


            model.load_initial_weights(sess)

            # print('pre train pruning info')
            # prune_info(weights_new, 0)
            # print(78*'-')
            # start = time.time()
            if TRAIN == 1:
                print("{} Start training...".format(datetime.now()))
                for i in range(0,epochs):
                    for step in range(train_batches_per_epoch):
                        (batch_x, batch_y) = train_generator.next_batch(batch_size)

                        train_acc, cross_en = sess.run([accuracy, loss_value], feed_dict = {
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: 1.0})

                        if (i % DISPLAY_FREQ == 0):
                            print('This is the {}th of {}pruning, time is {}'.format(
                                i,
                                cRates,
                                datetime.now()
                            ))
                            print("accuracy is {} and cross entropy is {}".format(
                                train_acc,
                                cross_en
                            ))
                            accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:19]))
                            if (i%(DISPLAY_FREQ*50) == 0 and i != 0 ):
                                train_acc_list.append(train_acc)
                                # file_name_part = compute_file_name(cRates)
                                # save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
                                # print("saved the network")
                            if (np.mean(accuracy_list) > 0.81 and train_acc >= 0.83):
                                accuracy_list = np.zeros(20)
                                test_acc = sess.run(accuracy, feed_dict = {
                                                        x: images_test,
                                                        y: labels_test,
                                                        keep_prob: 1.0})
                                print('test accuracy is {}'.format(test_acc))
                                if (test_acc > 0.823):
                                    print("training accuracy is large, show the list: {}".format(accuracy_list))
                                    break

                        _ = sess.run(train_step, feed_dict = {
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: dropout})

            if (TEST):
                test_acc_list = []

                if (LOCAL_TEST == 1):
                    image_dir = "cpu_test_data/tmp_images/"
                    imagenet_mean = np.array([104., 117., 124.], dtype = np.float32)
                    test_batches_per_epoch = 3
                    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]
                    imgs_test = []
                    for i,f in enumerate(img_files):
                        tmp = cv2.imread(f)
                        tmp = cv2.resize(tmp.astype(np.float32), (227,227))
                        tmp -= imagenet_mean[i]
                        tmp = tmp.reshape((1,227,227,3))
                        imgs_test.append(tmp)



                if (LOCAL_TEST):
                    names = []
                    probs = []

                    for step in range(test_batches_per_epoch):
                        prob = sess.run(softmax, feed_dict = {
                                                x: imgs_test[step],
                                                # y: labels_test,
                                                keep_prob: 1.0})
                        name = class_names[np.argmax(prob)]
                        probs.append(np.max(prob))
                        names.append(name)
                    print("names are {}".format(names))
                    print("probs are {}".format(probs))
                    sys.exit()
            # if (save_for_next_iter):
            #     print('saving for the next iteration of dynamic surgery')
            #     file_name_part = compute_file_name(cRates)
            #     file_name = 'weights'+ file_name_part+'.pkl'
            #     save_pkl_model(weights, biases, parent_dir, file_name)
            #
            #     file_name_part = compute_file_name(cRates)
            #     with open(parent_dir + 'mask' + file_name_part + '.pkl','wb') as f:
            #         pickle.dump((weights_mask, biases_mask),f)
            # if (TRAIN):
            #     file_name_part = compute_file_name(cRates)
            #     save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
            #     with open(parent_dir + 'training_data'+file_name_part+'.pkl', 'wb') as f:
            #         pickle.dump(train_acc_list, f)
            #
            # if (PRUNE):
            #     print('saving pruned model ...')
            #     f_name = compute_file_name(cRates)
            #     prune_weights(  cRates,
            #                     weights,
            #                     weights_mask,
            #                     biases,
            #                     biases_mask,
            #                     mask_dir,
            #                     'mask' + f_name + '.pkl')
            #     file_name_part = compute_file_name(cRates)
            #     save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
            # return test_acc
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return
if __name__ == '__main__':
    sys.exit(main())
