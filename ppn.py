from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


ppn = Perceptron(eta = 1 , n_iter=10)

ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.ylabel('Number of misclassifications')
plt.show()