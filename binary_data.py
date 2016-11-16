import random as rd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def get_data(n_samples):
        # makes binary classification data

        x_1 = [rd.normalvariate(0.7,0.2) for i in range(n_samples)]
        y_1 = [rd.normalvariate(0.2,0.2) for i in range(n_samples)]

        x_0 = [rd.normalvariate(0.2,0.2) for i in range(n_samples)]
        y_0 = [rd.normalvariate(0.7,0.2) for i in range(n_samples)]

        x = x_1 + x_0                
        y = y_1 + y_0

        labels = [1]*n_samples + [0]*n_samples

        features = [(i,j) for i,j in zip(x,y)]        
        indices = range(2*n_samples)
        rd.shuffle(indices)

        features_shuf = [features[i] for i in indices]
        labels_shuf   = [labels[i] for i in indices]

        return features_shuf, labels_shuf


if name == '__main__':

        feat, lab = get_data(100)

        col = ['g', 'r']

        for i in range(len(lab)):

                plt.scatter(feat[i][0], feat[i][1], color=col[lab[i]])

        plt.show()

