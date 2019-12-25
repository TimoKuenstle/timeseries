# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


__author__ = 'njkim@jamonglab.com'

# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 100   # batch size
num_category = 10  # category variable number
num_cont = 8   # continuous variable number
num_dim = 30   # total latent dimension ( category + continuous + noise )


#
# inputs
#

# target_number
target_num = tf.placeholder(dtype=tf.sg_intx, shape=batch_size)
# target continuous variable # 1
target_cval_1 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
# target continuous variable # 2
target_cval_2 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_3 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_4 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_5 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_6 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_7 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_8 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)



# category variables
z = (tf.ones(batch_size, dtype=tf.sg_intx) * target_num).sg_one_hot(depth=num_category)

# continuous variables
z = z.sg_concat(target=[target_cval_1.sg_expand_dims(), target_cval_2.sg_expand_dims(), target_cval_3.sg_expand_dims(), 
                        target_cval_4.sg_expand_dims(), target_cval_5.sg_expand_dims(), target_cval_6.sg_expand_dims(),
                        target_cval_7.sg_expand_dims(), target_cval_8.sg_expand_dims()])

# random seed = categorical variable + continuous variable + random uniform
z = z.sg_concat(target=tf.random_uniform((batch_size, num_dim-num_cont-num_category)))


#
# create generator
#

# generator network
with tf.sg_context(name='generator', size=(4, 1), stride=(2, 1), act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=48*1*128)
           .sg_reshape(shape=(-1, 48, 1, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=32)
           .sg_upconv(dim=num_cont, act='sigmoid', bn=False))#.sg_squeeze())


#
# run generator
#
def run_generator(num, x1, x2, x3, x4, x5, x6, x7, x8, fig_name='sample.png', csv_name='sample.csv'):
    with tf.Session() as sess:
        tf.sg_init(sess)
        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

        # run generator
        imgs = sess.run(gen, {target_num: num,
                              target_cval_1: x1,
                              target_cval_2: x2,
                              target_cval_3: x3,
                              target_cval_4: x4,
                              target_cval_5: x5,
                              target_cval_6: x6,
                              target_cval_7: x7,
                              target_cval_8: x8})
        print(imgs.shape)

        #imgs ist numpy.ndarray

        # plot result
        _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
        for i in range(10):
            for j in range(10):
                ax[i][j].plot(imgs[i * 10 + j, :, 0])
                pd.DataFrame(imgs[i * 10 + j, :, 0]).to_csv("asset/train/" + csv_name)

                #ax[i][j].plot(imgs[i * 10 + j, :, 1])
                ax[i][j].set_axis_off()
        plt.savefig('asset/train/' + fig_name, dpi=600)
        tf.sg_info('Sample image saved to "asset/train/%s"' % fig_name)
        tf.sg_info('Sample csv saved to "asset/train/%s"' % csv_name)
        plt.close()


#
# draw sample by categorical division
#

# fake image
run_generator(np.random.randint(0, num_category, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              fig_name='fake.png', csv_name='fake.csv')

# classified image
run_generator(np.arange(num_category).repeat(num_category),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size))

#
# draw sample by continuous division
#

for i in range(10):
    run_generator(np.ones(batch_size) * i,
                  np.linspace(0, 1, num_category).repeat(num_category),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  fig_name='sample%d.png' % i, csv_name='sample%d.csv' % i)
