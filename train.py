# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np


class TimeSeriesData(object):

    def __init__(self, batch_size=128):

        #Laden des Trainingsdatensatzes
        x = np.genfromtxt('asset/data/sample.csv', delimiter=',', dtype=np.float32) 
        x = x[1:, 1:]

        window = 384  # Fenstergröße, gibt unter anderem die Anzahl an generierten Beobachtungen an.  
       
        #Normieren der Daten
        max = np.amax(x) #Maximalwert über alle Daten bestimmen
        print("Maximum", max)

        # Nullzeilen löschen und Datenformat anpassen, so dass ein vielfaches des
        # Fenstergröße erzeugt wird.
        n = ((np.where(np.any(x, axis=1))[0][-1] + 1) // window) * window 

        # Daten zwischen 0 und 1 normieren um die Leistungsfähigkeit des GANs zu optimieren 
        #und die Rechenzeiten zu minimieren
        x = x[:n] / max
        print(x)

        # Zeitreihe in Matrixform bringen und zufällige Werte ziehen
        X = np.asarray([x[i:i+window] for i in range(n-window)])
        print(X.shape)
        np.random.shuffle(X)
        X = np.expand_dims(X, axis=2)

        self.batch_size = batch_size #Input Batchsize übernehmen
        self.X = tf.sg_data._data_to_tensor([X], batch_size, name='train')
        self.num_batch = X.shape[0] // batch_size


        self.X = tf.to_float(self.X)



tf.sg_verbosity(10)

# Hyper Parameter bestimmen

batch_size = 128   # Batchsize bestimmen 
num_category = 10  # Anzahl an kotegorischen Variablen definieren 
num_cont = 3   # Anzahl der zu erzeugenden Zeitreihen 
num_dim = 50   # Anzahl an latenten Dimensionen 
max_ep = 100   # Anzahl an Trainingsepochen

# Inputs

# Input tensor 
data = TimeSeriesData(batch_size=batch_size)
x = data.X

# Generator Labels (alle=1)
y = tf.ones(batch_size, dtype=tf.sg_floatx)

# Diskriminator Labels (1 und 0)
y_disc = tf.concat([y, y * 0], 0)



# Generator

z_cat = tf.multinomial(tf.ones((batch_size, num_category), dtype=tf.sg_floatx) / num_category, 1).sg_squeeze()

# Zufälliger Seed 
z = z_cat.sg_one_hot(depth=num_category).sg_concat(target=tf.random_uniform((batch_size, num_dim-num_category)))

# zufällige stetige Variable
z_cont = z[:, num_category:num_category+num_cont]

# Definieren des Generatornetzwerkes 
with tf.sg_context(name='generator', size=(4, 1), stride=(2, 1), act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=window / 8*1*128) #Fenstergröße muss in diesem Fall durch 8 teilbar sein
           .sg_reshape(shape=(-1, window/8, 1, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=32)
           .sg_upconv(dim=num_cont, act='sigmoid', bn=False))  #Output Dimension der Spalten entspricht der Anzahl des Lerndatensatzes


# Definieren des Diskriminatornetzwerkes

print x
print  gen
# Echte und Falsche Bilder erzeugen, die der Diskriminator nutzt um Daten zu klassifizeren
xx = tf.concat([x, gen], 0)

with tf.sg_context(name='discriminator', size=(4, 1), stride=(2, 1), act='leaky_relu'):
    shared = (xx.sg_conv(dim=32)
              .sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_flatten()
              .sg_dense(dim=1024)
    recog_shared = shared[batch_size:, :].sg_dense(dim=128)           
    disc = shared.sg_dense(dim=1, act='linear').sg_squeeze()
    recog_cat = recog_shared.sg_dense(dim=num_category, act='linear')
    recog_cont = recog_shared.sg_dense(dim=num_cont, act='sigmoid')

# Verlustfunktionen des Netztwerkes definieren. Diese sollen minimer werden

loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # Discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=gen).sg_bce(target=y))  # Generator loss
loss_recog = tf.reduce_mean(recog_cat.sg_ce(target=z_cat)) \
             + tf.reduce_mean(recog_cont.sg_mse(target=z_cont))  # Recognizer loss

train_disc = tf.sg_optim(loss_disc + loss_recog, lr=0.0001, category='discriminator')  # Discriminator train ops
train_gen = tf.sg_optim(loss_gen + loss_recog, lr=0.001, category='generator')  # Generator train ops


# Trainingsprozess des GANs
              
#Definiere Trainingsfunktion              
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # Training des Diskriminators
    l_gen = sess.run([loss_gen, train_gen])[0]  #Training des Generators
    return np.mean(l_disc) + np.mean(l_gen)


#Anzahl der Trainingsepochen bestimmen
alt_train(log_interval=10, max_ep=max_ep, ep_size=data.num_batch, early_stop=False)

