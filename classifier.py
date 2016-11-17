from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import binary_data as bd
import tensorflow as tf
import net as net

def ANN(pt):

        l0 = tf.nn.softplus(net.feed_forward(pt, 10, 'adversary', 'l0'))
        l1 = net.feed_forward(l0, 2, 'adversary', 'l2')

        return l1

def run_net(train, test):

        X = tf.placeholder(tf.float32, [1, 2], name="X")
        Y = tf.placeholder(tf.float32, [1, 2], name="Y")

        y_c = ANN(X)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_c, Y))

        train_op = tf.train.AdamOptimizer().minimize(cost)
        test_op = tf.equal(tf.argmax(y_c, 1), tf.argmax(Y, 1))

        with tf.Session() as sess:

                sess.run(tf.initialize_all_variables())
                for obs in train:
                        pt = np.array([obs[:2]])
                        label = np.zeros((1,2))
                        label[0][int(obs[2])] += 1

                        sess.run(train_op, feed_dict={X:pt, Y:label})
        
                guess = []
                for obs in test:
                        pt = np.array([obs[:2]])
                        label = np.zeros((1,2))
                        label[0][int(obs[2])] += 1

                        guess.append(
                  sess.run(test_op, feed_dict={X:pt, Y:label})[0])
                
        return sum(guess) / float(len(guess))     


def supp_vec(train, test):

        model = SVC()
        X = train[:,:2]
        Y = train[:,2]
        
        model.fit(X,Y)

        acc = accuracy_score(test[:,2], model.predict(test[:,:2]))
        return acc
 
def logit(train, test):

        model = LogisticRegression()
        X = train[:,:2]
        Y = train[:,2]
        
        model.fit(X, Y)

        acc = accuracy_score(test[:,2], model.predict(test[:,:2]))

        return acc


if __name__ == '__main__':

        n = 1000
        data, label = bd.get_data(n)

        train_data = data[:int(0.9*n)]
        test_data  = data[int(0.9*n):]
        train_label = label[:int(0.9*n)]
        test_label  = label[int(0.9*n):]

        train = np.hstack((train_data, np.array([train_label]).T))
        test  = np.hstack((test_data, np.array([test_label]).T))

        svm_pc_acc   = 100*supp_vec(train, test)
        logit_pc_acc = 100*logit(train, test)

        nn_acc = run_net(train, test)

        print(
" Logistic Regression Acc: %.2f%% \n \
SVM Acc :                %.2f%% \n \
Neural Net Acc:          %.2f%%" \
%(logit_pc_acc, svm_pc_acc, nn_acc*100)
)

