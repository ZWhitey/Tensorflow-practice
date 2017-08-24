import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

#Generate Data
x=np.linspace(-1,1,100)[:,np.newaxis]
noisy=np.random.normal(0,0.1,x.shape)
y=2*np.power(x,4)+2*np.power(x,3)-2*x+noisy

#Preview Data
plt.scatter(x,y)
plt.show()

#Input Layer
with tf.name_scope('Input'):
    tfX=tf.placeholder(tf.float32,x.shape)
tfY=tf.placeholder(tf.float32,y.shape)

#Hidden Layer
l1=tf.layers.dense(tfX,10,tf.nn.relu,name="Layer")

#Output Layer
output=tf.layers.dense(l1,1,name="Output")

#Compute Loss
with tf.name_scope('Loss'):
    loss=tf.losses.mean_squared_error(tfY,output)

#Training
with tf.name_scope('Train'):
    opt=tf.train.GradientDescentOptimizer(0.5)
    train=opt.minimize(loss)
#Init Variable
init=tf.global_variables_initializer()

feed={tfX:x,tfY:y}
plt.ion()

#Learning
with tf.Session() as sess:
    sess.run(init)
    train_writer=tf.summary.FileWriter('log/regression',sess.graph)
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