import tensorflow as tf

def showConstant(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    sess.close()

def showConstantDetail(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    print('Shape: ', tf.shape(t))
    print('Size: ', tf.size(t))
    print('Rank: ', tf.rank(t))
    print(t.get_shape())
    sess.colse()

def showVariable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    print(v.eval())
    sess.close()