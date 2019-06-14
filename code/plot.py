import matplotlib.pyplot as plt
import numpy as np

def plot_classification_error(train_errors, test_errors, labels, dataset):

    # Sort by test error

    sorted_indices = np.flipud(np.argsort(test_errors))

    test_errors = test_errors[sorted_indices]
    train_errors = train_errors[sorted_indices]
    labels = labels[sorted_indices]

    pos = np.arange(len(labels))
    bar_width = 0.35
    error_legend = ["Training Error", "Test Error"]

    # Plot the training and test error bars

    plt.barh(pos, train_errors, bar_width, color='gray', edgecolor='black')
    plt.barh(pos+bar_width, test_errors, bar_width, color='green', edgecolor='black')

    plt.yticks(pos, labels)
    plt.ylabel('Model')
    plt.xlabel('Error Rate')
    plt.legend(error_legend, bbox_to_anchor=(1.05, 1), loc=2)

    if dataset == "1":
        plt.title('Breast Tissue Classification Error')
    
    filename = '../figs/classification_error_%s.png' % (dataset)
    plt.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    print('\nFigure saved as %s' % filename)
    
    return None

def plot_visualization(Z, y, title, filename):

    colors = ['red' if y_i == 1 else 'gray' for y_i in y]
    fig, ax = plt.subplots()

    ax.scatter(Z[y==0,0], Z[y==0,1], color='gray', label='Benign')
    ax.scatter(Z[y==1,0], Z[y==1,1], color='red', label='Malignant')

    plt.ylabel('z2')
    plt.xlabel('z1')
    plt.title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

    filename = '../figs/%s.png' % (filename)
    plt.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    print('\nFigure saved as %s' % filename)
    
    return None
