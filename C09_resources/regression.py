import numpy as np 
import tensorflow as tf 

class Error(Exception):
    pass

class NotFittedError(Error):
    pass

class ShapesNotCompatibleError(Error):
    pass

class DataBatch:
    def __init__(self, batch_size):
        self._batch_size = batch_size

    def batch(self, X, y, epoch):
        n_examples = X.shape[0]
        n_batches = int( np.ceil( n_examples / self._batch_size ) )

        for batch_index in range(n_batches):
            np.random.seed(epoch * n_batches + batch_index)
            indices = np.random.randint(n_examples, size=self._batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            yield X_batch, y_batch

class LinearRegression:
    def __init__(self, learning_rate=1e-2, batch_size=100, n_epochs=2000, print_every=200):
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._print_every = print_every
        self._theta = None

    def fit(self, X, y):
        self._n_features = X.shape[1]
        data = DataBatch(self._batch_size)

        with tf.Graph().as_default():
            # placeholders and variables
            tX = tf.placeholder(tf.float32, shape=(None, self._n_features), name='X')
            ty = tf.placeholder(tf.float32, shape=(None, 1), name='y')
            theta = tf.Variable(tf.zeros([self._n_features, 1]), name='theta')
            
            # initialization step
            init = tf.global_variables_initializer()

            # reducing to canonical form
            XT = tf.transpose(tX)
            H = tf.matmul(XT, tX)
            b = tf.matmul(XT, ty)
            c = tf.matmul(tf.transpose(ty), ty)

            # evaluate mse
            n_examples = tf.shape(X)[0]
            fval = tf.matmul(tf.math.add(tf.matmul(tf.transpose(theta), H), - 2.0 * tf.transpose(b)), theta) + c
            mse = tf.squeeze(tf.divide(fval, tf.cast(n_examples, tf.float32)), name='mse')

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            training_op = optimizer.minimize(mse, name='training_op')
            
            # summary
            mse_summary = tf.summary.scalar('MSE', mse)
            file_writer = tf.summary.FileWriter('tmp', tf.get_default_graph())

            saver = tf.train.Saver()
            with tf.Session() as sess:
                
                sess.run(init)
                for epoch in range(self._n_epochs):
                    for X_batch, y_batch in data.batch(X, y, epoch):
                        _, mse_batch = sess.run([training_op, mse], feed_dict={tX: X_batch, ty: y_batch} )

                    if epoch % self._print_every == 0 or epoch == self._n_epochs-1:
                        print('Epoch {: 5d}, MSE {:4.2e}'.format(epoch, mse_batch))
                        
                        save_path = saver.save(sess, 'tmp/model.ckpt')
                        file_writer.add_summary(mse_summary.eval(feed_dict={tX: X_batch, ty: y_batch}), epoch)

                self._theta = theta.eval()
                save_path = saver.save(sess, 'tmp/model_final.ckpt')
            
                file_writer.close()
        return self
    
    def predict(self, X):
        if self._theta is None:
            raise NotFittedError()
        
        if X.shape[1] != self._n_features:
            raise ShapesNotCompatibleError()
        
        graph = tf.Graph()
        with graph.as_default():
            tX = tf.placeholder(tf.float32, shape=(None, self._n_features), name='X')
            theta = tf.placeholder(tf.float32, shape=(self._n_features, 1), name='theta')
            pred = tf.matmul(tX, theta)

            with tf.Session() as sess:
                predictions = sess.run(pred, feed_dict={tX: X, theta: self._theta})

        return predictions


class LogisticRegression:
    def __init__(self, learning_rate=1e-2, batch_size=100, n_epochs=2000, print_every=200):
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._print_every = print_every
        self._theta = None

    def fit(self, X, y):
        self._n_features = X.shape[1]
        data = DataBatch(self._batch_size)

        with tf.Graph().as_default():
            # placeholders and variables
            tX = tf.placeholder(tf.float32, shape=(None, self._n_features), name='X')
            ty = tf.placeholder(tf.float32, shape=(None, 1), name='y')
            theta = tf.Variable(tf.zeros([self._n_features, 1]), name='theta')
            
            # initialization step
            init = tf.global_variables_initializer()

            # evaluate probabilities
            predicted_proba = tf.math.sigmoid(tf.matmul(tX, theta), 'predicted_proba')
            
            loss = tf.losses.log_loss(ty, predicted_proba)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            training_op = optimizer.minimize(loss, name='training_op')            

            # summary
            loss_summary = tf.summary.scalar('loss', loss)
            file_writer = tf.summary.FileWriter('tmp', tf.get_default_graph())
            saver = tf.train.Saver()   

            with tf.Session() as sess:    
                sess.run(init)
                for epoch in range(self._n_epochs):
                    for X_batch, y_batch in data.batch(X, y, epoch):
                        _, loss_batch = sess.run([training_op, loss], feed_dict={tX: X_batch, ty: y_batch} )

                    if epoch % self._print_every == 0 or epoch == self._n_epochs-1:
                        print('Epoch {: 5d}, loss {:4.2e}'.format(epoch, loss_batch))
                        
                        save_path = saver.save(sess, 'tmp/model.ckpt')
                        file_writer.add_summary(loss_summary.eval(feed_dict={tX: X_batch, ty: y_batch}), epoch)

                self._theta = theta.eval()
                save_path = saver.save(sess, 'tmp/model_final.ckpt')
            
                file_writer.close()
        return self
    
    def predict(self, X):
        if self._theta is None:
            raise NotFittedError()
        
        if X.shape[1] != self._n_features:
            raise ShapesNotCompatibleError()
        
        graph = tf.Graph()
        with graph.as_default():
            tX = tf.placeholder(tf.float32, shape=(None, self._n_features), name='X')
            theta = tf.placeholder(tf.float32, shape=(self._n_features, 1), name='theta')
            predicted_proba = tf.math.sigmoid(tf.matmul(tX, theta), 'predicted_proba')

            classes = tf.cast(predicted_proba + 0.5, tf.int32)

            with tf.Session() as sess:
                predictions = sess.run(classes, feed_dict={tX: X, theta: self._theta})

        return predictions