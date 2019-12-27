# Pakete Laden
import sugartensor as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


tf.sg_verbosity(10)

# Hyper Parameter bestimmen

batch_size = 100   # Batchsize
num_category = 10  # Anzahl der kategorischen Variablen
num_cont = 3   # Anzahl der zu erzeugenden Zeitreihen
num_dim = 50   # Anzahl der latenten Dimensionen 
window = 376 # Analog zu train.py. Darf maximal (Anzahl Zeilen der Trainingsdatei / 2) - 1 sein.


# Inputs

target_num = tf.placeholder(dtype=tf.sg_intx, shape=batch_size)
target_cval_1 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_2 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
target_cval_3 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size) 
# Anzahl der Variablen ist abhängig von der Anzahl der zu erzeugenden Zeitreihen.


# Kategorischen Variablen erzeugen
z = (tf.ones(batch_size, dtype=tf.sg_intx) * target_num).sg_one_hot(depth=num_category)

# Zeitreihenvariabln erzeugen, Anzahl der Variablen ist abhängig von der Anzahl der zu erzeugenden Zeitreihen.
z = z.sg_concat(target=[target_cval_1.sg_expand_dims(), target_cval_2.sg_expand_dims(), target_cval_3.sg_expand_dims()])

# Zufälliger Seed
z = z.sg_concat(target=tf.random_uniform((batch_size, num_dim-num_cont-num_category)))

# Generator Netzwerk (Analog zu train.py)
with tf.sg_context(name='generator', size=(4, 1), stride=(2, 1), act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=window/8*1*128)
           .sg_reshape(shape=(-1, window/8, 1, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=32)
           .sg_upconv(dim=num_cont, act='sigmoid', bn=False))

#Generator ausführen
#Dafür erst Generator aufrufen und und Spalten definieren
#Anzahl der Variablen ist abhängig von der Anzahl der zu erzeugenden Zeitreihen.

def run_generator(num, x1, x2, x3, fig_name='sample.png', csv_name='sample.csv'):
    with tf.Session() as sess:
        tf.sg_init(sess)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

        # Generator ausführen mit Zielwerte, wo die Zeitreihen erzeugt werden. #target_cval = #Zeitreihen
        imgs = sess.run(gen, {target_num: num,
                              target_cval_1: x1,
                              target_cval_2: x2,
                              target_cval_3: x3})
        print(imgs.shape)

        #imgs ist ein numpy.ndarray

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


# Ziehen von Stichproben der erzeugten Daten

# Anzahl der Argumente ist abhängig von der Anzahl der zu erzeugenden Zeitreihen.
# gefälschtes Bild
run_generator(np.random.randint(0, num_category, batch_size),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size),
              fig_name='fake.png', csv_name='fake.csv')

# klassifiziertes Bild
run_generator(np.arange(num_category).repeat(num_category),
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              np.random.uniform(0, 1, batch_size))


# Stichprobe ziehen

for i in range(10):
    run_generator(np.ones(batch_size) * i,
                  np.linspace(0, 1, num_category).repeat(num_category),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  np.expand_dims(np.linspace(0, 1, num_category), axis=1).repeat(num_category, axis=1).T.flatten(),
                  fig_name='sample%d.png' % i, csv_name='sample%d.csv' % i)
