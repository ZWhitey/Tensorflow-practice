import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:,np.newaxis]
noisy=np.random.normal(0,0.1,x.shape)
y=2*np.power(x,4)+2*np.power(x,3)-2*x+noisy

plt.scatter(x,y)
plt.show()

tfX=tf.placeholder(tf.float32,x.shape)
tfY=tf.placeholder(tf.float32,y.shape)
l1=tf.layers.dense(tfX,10,tf.nn.relu)
output=tf.layers.dense(l1,1)
loss=tf.losses.mean_squared_error(tfY,output)
opt=tf.train.GradientDescentOptimizer(0.5)
train=opt.minimize(loss)
init=tf.global_variables_initializer()

feed={tfX:x,tfY:y}
plt.ion()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        _,l,pred=sess.run([train,loss,output],feed_dict=feed)
        if i%50==0:
            plt.cla()
            plt.scatter(x,y)
            plt.plot(x,pred,'r-',lw=5)
            plt.text(0.5,0,'Loss=%.4f'%l)
            plt.pause(0.1)
    plt.ioff()
    plt.show()