import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE

import data
import plot

def hyperparameter_search(model, param, X, y):
    cv = GridSearchCV(model, param, iid=False, cv=5, scoring='accuracy')
    cv.fit(X, y)

    return cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', required=False)
    io_args = parser.parse_args()
    dataset = io_args.dataset

    if dataset == "1" or dataset == None:

        # Load dataset 1, split into training and test sets
        X, y, Xtest, ytest = data.load_training_and_test_data("breast-cancer-wisconsin-data.csv")

        # Perform Visualization with PCA, MDS, ISOMAP, T-SNE
        X = np.concatenate((X,Xtest),axis=0)
        y = np.concatenate((y,ytest),axis=0)

        model = PCA(n_components=2)
        Z = model.fit_transform(X)
        plot.plot_visualization(Z,y,'Breast Tissue Dataset - PCA','pca_%s' % dataset)

        model = MDS(n_components=2)
        Z = model.fit_transform(X)
        plot.plot_visualization(Z,y,'Breast Tissue Dataset - MDS','mds_%s' % dataset)  
        
        model = Isomap(n_neighbors=5, n_components=2)
        Z = model.fit_transform(X)
        plot.plot_visualization(Z,y,'Breast Tissue Dataset - ISOMAP','isomap_%s' % dataset) 

        model = TSNE(n_components=2)
        Z = model.fit_transform(X)
        plot.plot_visualization(Z,y,'Breast Tissue Dataset - t-SNE','tsne_%s' % dataset)


        # Compare ML algorithms on dataset 1
        k_times = 5

        model_names = ['Random Forest Classifier',
                        'K Neighbors Classifier',
                        'Gaussian Naive Bayes Classifier',
                        'Support Vector Machine',
                        'MLP Classifier']

        test_errors = np.zeros(5)
        train_errors = np.zeros(5)

        for i in range(k_times):
            # Load dataset 1, split into training and test sets
            X, y, Xtest, ytest = data.load_training_and_test_data("breast-cancer-wisconsin-data.csv")

            # RANDOM FOREST

            param = {'n_estimators': [10,20,30,40,50,60,70,80,90,100]}
            cv = hyperparameter_search(RandomForestClassifier(), param, X, y)
                    
            rf = RandomForestClassifier(n_estimators=cv.best_params_['n_estimators'])
            rf.fit(X,y)
            
            train_error_rf = np.mean(rf.predict(X) != y)
            test_error_rf = np.mean(rf.predict(Xtest) != ytest)

            print("\nRandom Forest Classifier")
            print("n_estimators: %s" % rf.n_estimators)
            print("Training Error: %0.3f" % train_error_rf)
            print("Test Error: %0.3f" % test_error_rf)

            test_errors[0] += test_error_rf
            train_errors[0] += train_error_rf

            # KNN

            param = {'n_neighbors': range(1,11)}
            cv = hyperparameter_search(KNeighborsClassifier(), param, X, y)

            knn = KNeighborsClassifier(n_neighbors=cv.best_params_['n_neighbors'])
            knn.fit(X,y)

            train_error_knn = np.mean(knn.predict(X) != y)
            test_error_knn = np.mean(knn.predict(Xtest) != ytest)

            print("\nK Neighbors Classifier")
            print("n_neighbors: %s" % knn.n_neighbors)
            print("Training Error: %0.3f" % train_error_knn)
            print("Test Error: %0.3f" % test_error_knn)

            test_errors[1] += test_error_knn
            train_errors[1] += train_error_knn

            # NAIVE BAYES

            param = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
            cv = hyperparameter_search(GaussianNB(), param, X, y)

            nb = GaussianNB(var_smoothing=cv.best_params_['var_smoothing'])
            nb.fit(X,y)

            train_error_nb = np.mean(nb.predict(X) != y)
            test_error_nb = np.mean(nb.predict(Xtest) != ytest)

            print("\nGaussian NB")
            print("var_smoothing: %s" % nb.var_smoothing)
            print("Training Error: %0.3f" % train_error_nb)
            print("Test Error: %0.3f" % test_error_nb)

            test_errors[2] += test_error_nb
            train_errors[2] += train_error_nb

            # SVM

            param = {'C': [1e0, 1e1, 1e2, 1e3, 1e4]}
            cv = hyperparameter_search(SVC(gamma='scale'), param, X, y)

            svm = SVC(C=cv.best_params_['C'], gamma='scale')
            svm.fit(X,y)

            train_error_svm= np.mean(svm.predict(X) != y)
            test_error_svm = np.mean(svm.predict(Xtest) != ytest)

            print("\nSupport Vector Machine")
            print("C: %s" % svm.C)
            print("Training Error: %0.3f" % train_error_svm)
            print("Test Error: %0.3f" % test_error_svm)

            test_errors[3] += test_error_svm
            train_errors[3] += train_error_svm

            # NEURAL NETWORK

            param = {'hidden_layer_sizes': [(10,),(20,),(30,),(40,),(50,)],
                        'alpha': [1e-5,1e-4,1e-3,1e-2,1e-1]}
            cv = hyperparameter_search(MLPClassifier(max_iter=200), param, X, y)

            nnet = MLPClassifier(hidden_layer_sizes=cv.best_params_['hidden_layer_sizes'], alpha=cv.best_params_['alpha'], max_iter=200)
            nnet.fit(X,y)

            train_error_nnet = np.mean(nnet.predict(X) != y)
            test_error_nnet = np.mean(nnet.predict(Xtest) != ytest)

            print("\nMLP Classifier")
            print("hidden_layer_sizes: %s" % nnet.hidden_layer_sizes)
            print("alpha: %s" % nnet.alpha)
            print("Training Error: %0.3f" % train_error_nnet)
            print("Test Error: %0.3f" % train_error_nnet)
            
            test_errors[4] += test_error_nnet
            train_errors[4] += train_error_nnet

        # Compute the averages

        test_errors = test_errors / k_times
        train_errors = train_errors / k_times

        # Summarize Results

        print("\nAverage Results from %s trials:" % k_times)

        for i, name in enumerate(model_names):
            print("\n%s" % name)
            print("Training Error: %0.3f" % train_errors[i])
            print("Test Error: %0.3f" % test_errors[i])

        # Plot average classification error of each model
        labels = np.array(['RF', 'KNN', 'NB', 'SVM', 'MLP'])
        plot.plot_classification_error(train_errors, test_errors, labels, dataset)

    elif dataset == "2":
        # TODO load dataset 2, split into training and test sets
        # breast-cancer-coimbra-data.csv

        # TODO compare ML algorithms on dataset 2

        # TODO generate comparison plot of test error of each model

        pass
    
    else:
        print("Unknown dataset %s" % dataset)

