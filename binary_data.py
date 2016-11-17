import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def get_data(n_samples):
        # makes binary classification data
        x_train_1 = np.random.normal(0.7, 0.2, n_samples)
        y_train_1 = np.random.normal(0.2, 0.2, n_samples)

        x_train_0 = np.random.normal(0.2, 0.2, n_samples)
        y_train_0 = np.random.normal(0.7, 0.2, n_samples)

        x = np.append(x_train_1, x_train_0)
        y = np.append(y_train_1, y_train_0)
        pts = np.asarray([x, y]).T
        
        labels = np.append(np.ones(n_samples), np.zeros(n_samples))
        indices = range(2*n_samples)
        np.random.shuffle(indices)
        pts_shuf = pts[indices]
        lab_shuf = labels[indices]

        return pts_shuf, lab_shuf


if __name__ == '__main__':

        feat, lab = get_data(100)


        plt.scatter(feat[:,0][np.where(lab == 1.)], 
                   feat[:,1][np.where(lab == 1.)], color='g')
        
        plt.scatter(feat[:,0][np.where(lab == 0.)], 
                   feat[:,1][np.where(lab == 0.)], color='r')


        plt.show()
