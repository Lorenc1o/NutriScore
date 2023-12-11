import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

def values_to_scale(dataset_location, column, intervals, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)

    # Get the values of the specified column
    values = dataset[column]

    # Calculate the scale
    scale = np.zeros(len(values))

    '''
        if column == 'perc_fruit':
        # The case of the percentage of fruits and vegetables is different
        # 0-40%: 0
        # 40-60%: 1
        # 60-80%: 2
        # 80-100%: 5
        intervals = [40,60]

        j = 0
        for value in values:
            if value >= 80:
                scale[j] = 5
            else:
                for i, interval in enumerate(intervals):
                    if value <= interval:
                        scale[j] = i
                        break

            j += 1
    '''

    j = 0
    for value in values: 
        # For each value, find the interval it belongs to and assign the corresponding value,
        # if the value is greater than the last interval, assign the last value
        for i, interval in enumerate(intervals):
            if value <= interval:
                scale[j] = i
                break
            elif i == len(intervals)-1:
                scale[j] = i+1        
        j += 1

    # Return the scale
    return scale

def additive_nutri_score(dataset_location, columns, utility_functions, weights, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)

    # Calculate the score
    scores = np.zeros(len(dataset))

    for i, column in enumerate(columns):
        scores += weights[i] * utility_functions[column](dataset[column])

    # Convert each score to a letter
    final = []
    for i, value in enumerate(scores):
        if value >= 16:
            final.append('a')
        elif value >= 12 and value < 16:
            final.append('b')
        elif value >= 8 and value < 12:
            final.append('c')
        elif value >= 6 and value < 8:
            final.append('d')
        elif value < 6:
            final.append('e')
    
    return scores, final

def show_scores_histogram(scores, bins=10):
    # Show the histogram of the scores
    plt.hist(scores, bins=bins)
    plt.show()

def show_confusion_matrix(new_scores, initial_scores, title, xlabel, ylabel):
    # Show the confusion matrix: a, b, c, d, e
    confusion_matrix = np.zeros((5,5))
    idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

    for i, score in enumerate(initial_scores):
        confusion_matrix[idx[score]][idx[new_scores[i]]] += 1

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(idx)))
    ax.set_yticks(np.arange(len(idx)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(idx.keys())
    ax.set_yticklabels(idx.keys())

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(idx)):
        for j in range(len(idx)):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w")
            
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Add a colorbar
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()

def compare_profiles(new, base, weights, lambda_threshold, direction):
    acc = 0
    for i, value in enumerate(new):
        if direction[i] == 'big':
            if value >= base[i]:
                acc += weights[i]
        else:
            if value <= base[i]:
                acc += weights[i]
    if acc >= lambda_threshold:
        return True
    return False

def PessimisticMajoritySorting(dataset_location, columns, utility_functions, weights, profiles, lambda_threshold, directions, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)
    dataset = dataset[columns]

    # Calculate the score
    nutriscores = []

    total_weights = sum(weights)

    # Create dataframe utilities with columns = columns, and rows = len(dataset)
    utilities = pd.DataFrame(columns=columns, index=range(len(dataset)))

    for column in columns:
        utilities[column] = utility_functions[column](dataset[column])

    for i, row in utilities.iterrows():
        if compare_profiles(row, profiles['a'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('a')
        elif compare_profiles(row, profiles['b'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('b')
        elif compare_profiles(row, profiles['c'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('c')
        elif compare_profiles(row, profiles['d'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('d')
        else:
            nutriscores.append('e')
    
    return nutriscores

def OptimisticMajoritySorting(dataset_location, columns, utility_functions, weights, profiles, lambda_threshold, directions, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)
    dataset = dataset[columns]

    # Calculate the score
    nutriscores = []

    total_weights = sum(weights)

    # Create dataframe utilities with columns = columns, and rows = len(dataset)
    utilities = pd.DataFrame(columns=columns, index=range(len(dataset)))

    for column in columns:
        utilities[column] = utility_functions[column](dataset[column])

    for i, row in utilities.iterrows():
        if not compare_profiles(row, profiles['d'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('e')
        elif not compare_profiles(row, profiles['c'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('d')
        elif compare_profiles(row, profiles['b'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('c')
        elif not compare_profiles(row, profiles['a'], weights, lambda_threshold*total_weights, directions):
            nutriscores.append('b')
        else:
            nutriscores.append('a')

    return nutriscores

def DecisionTreeNutriscore(dataset_location, columns, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)
    dataset = dataset[columns]

    # Separate the dataset into training and test
    # 60% training, 40% test
    training = dataset.sample(frac=0.6)
    test = dataset.drop(training.index)

    print('Training dataset:', training)

    # Train the decision tree
    X = training.drop(columns=['nutriscore'])
    y = training['nutriscore']

    clf = DecisionTreeClassifier()

    clf.fit(X, y)

    # Predict the nutriscore for the test dataset
    X_test = test.drop(columns=['nutriscore'])
    y_test = test['nutriscore']

    y_pred = clf.predict(X_test)

    return clf, clf.predict(dataset.drop(columns=['nutriscore'])), accuracy_score(y_test, y_pred)

def kNN_Nutriscore(dataset_location, columns, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)
    dataset = dataset[columns]

    # Separate the dataset into training and test
    # 60% training, 40% test
    training = dataset.sample(frac=0.6)
    test = dataset.drop(training.index)

    print('Training dataset:', training)

    # Train the kNN classifier
    X = training.drop(columns=['nutriscore'])
    y = training['nutriscore']

    clf = KNeighborsClassifier(n_neighbors=5)

    clf.fit(X, y)

    # Predict the nutriscore for the test dataset
    X_test = test.drop(columns=['nutriscore'])
    y_test = test['nutriscore']

    y_pred = clf.predict(X_test)

    return clf, clf.predict(dataset.drop(columns=['nutriscore'])), accuracy_score(y_test, y_pred)

def RandomForestNutriscore(dataset_location, columns, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)
    dataset = dataset[columns]

    # Separate the dataset into training and test
    # 60% training, 40% test
    training = dataset.sample(frac=0.6)
    test = dataset.drop(training.index)

    print('Training dataset:', training)

    # Train the random forest classifier
    X = training.drop(columns=['nutriscore'])
    y = training['nutriscore']

    clf = RandomForestClassifier()

    clf.fit(X, y)

    # Predict the nutriscore for the test dataset
    X_test = test.drop(columns=['nutriscore'])
    y_test = test['nutriscore']

    y_pred = clf.predict(X_test)

    return clf, clf.predict(dataset.drop(columns=['nutriscore'])), accuracy_score(y_test, y_pred)


filename = 'data.csv'

# Exercise 4 

columns_ex4 = [#'energy', 
    'fat_g', 'sugar_g', 'sodium_mg', 'perc_fruit', 'fibers_g', 'proteins_g']

# Changes:
# - Doubled the granularity of the utility functions
# - Fruit is considered very important, so we give max points with just 50% of fruit
# - Weights are now 1/6 for each attribute
# - The scale goes from 0 to 20, making it easier to comprehend

utility_functions = { 'energy': lambda x: 20-values_to_scale(filename, 'energy', [167.5,335,502.5,670,837.5,1005,1172.5,1340,1507.5,1675,1842.5,2010,2177.5,2345,2512.5,2680,2847.5,3015,3182.5,3350]),     # 65.2;6.6;20.0;100;14.8;7.4
                        'fat_g': lambda x: 20-values_to_scale(filename, 'fat_g', [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]), # 0
                        'sugar_g': lambda x: 20-values_to_scale(filename, 'sugar_g', [2.25,4.5,6.75,9,11.25,13.5,15.75,18,20.25,22.5,25.75,27,29.25,31,33.25,35.5,37.75,40,42.25,45]), # 18
                        'sodium_mg': lambda x: 20-values_to_scale(filename, 'sodium_mg', [45,90,135,180,225,270,315,360,405,450,495,540,585,630,675,720,765,810,855,900]), # 20
                        'perc_fruit': lambda x: 2*values_to_scale(filename, 'perc_fruit', [5,10,15,20,25,30,35,40,45,50]), # 20
                        'fibers_g': lambda x: 2*values_to_scale(filename, 'fibers_g', [0.47,0.94,1.42,1.89,2.36,2.83,3.3,3.77,4.24,4.7]), # 20
                        'proteins_g': lambda x: 2*values_to_scale(filename, 'proteins_g', [0.8,1.6,2.4,3.2,4,4.8,5.6,6.4,7.2,8])}  # 18

weights_ex4 = [#1/6,
           1/6,1/6,1/6,1/6,1/6,1/6]

# Exercise 5
columns_ex5 = ['energy', 'fat_g', 'sugar_g', 'sodium_mg', 'perc_fruit', 'fibers_g', 'proteins_g']
weights_ex5 = [1,1,1,1,2,1,1]
directions_ex5 = ['small','small','small','small','big','big','big']

lambda_ex5 = [0.5, 0.6, 0.7]

profiles = {
    'best': [20, 20, 20, 20, 20, 20, 20],
    'a': [16, 16, 16, 16, 16, 16, 16],
    'b': [12, 12, 12, 12, 12, 12, 12],
    'c': [8, 8, 8, 8, 8, 8, 8],
    'd': [6, 6, 6, 6, 6, 6, 6],
    'e': [0, 0, 0, 0, 0, 0, 0]
}

# Exercise 6
columns_ex6 = ['energy', 'fat_g', 'sugar_g', 'sodium_mg', 'perc_fruit', 'fibers_g', 'proteins_g', 'nutriscore']

if __name__ == '__main__':
    # Calculate the additive score
    additive_score, nutriscore = additive_nutri_score(filename, columns_ex4, utility_functions, weights_ex4)

    # Calculate the Pessimistic Majority Sorting
    pms = [PessimisticMajoritySorting(filename, columns_ex5, utility_functions, weights_ex5, profiles, lambda_threshold, directions_ex5) for lambda_threshold in lambda_ex5]

    # Calculate the Optimistic Majority Sorting
    oms = [OptimisticMajoritySorting(filename, columns_ex5, utility_functions, weights_ex5, profiles, lambda_threshold, directions_ex5) for lambda_threshold in lambda_ex5]

    # Calculate the Decision Tree Nutriscore
    clf, decision_tree_nutriscore, accuracy = DecisionTreeNutriscore(filename, columns_ex6)

    # Calculate the kNN Nutriscore
    clf_kNN, kNN_score, accuracy_kNN = kNN_Nutriscore(filename, columns_ex6)

    # Calculate the Random Forest Nutriscore
    clf_RF, RF_score, accuracy_RF = RandomForestNutriscore(filename, columns_ex6)

    # Print the computed score and the nutriscore from the dataset for each product
    dataset = pd.read_csv(filename, sep=';')
    #n_corrects = 0
    #for i, row in dataset.iterrows():
        # Print the two scores and the comparison of their lowcase versions
    #    print(row['nutriscore'], nutriscore[i], additive_score[i], row['nutriscore'].lower() == nutriscore[i].lower())
    #    if row['nutriscore'].lower() == nutriscore[i].lower():
    #        n_corrects += 1

    # Print the percentage of correct scores
    #print('Percentage of correct scores:', n_corrects / len(dataset))

    all_scores = {
        'nutriscore': dataset['nutriscore'],
        'G_0': nutriscore,
        'PMS_0.5': pms[0],
        'PMS_0.6': pms[1],
        'PMS_0.7': pms[2],
        'OMS_0.5': oms[0],
        'OMS_0.6': oms[1],
        'OMS_0.7': oms[2],
        'Decision Tree': decision_tree_nutriscore,
        'kNN': kNN_score,
        'Random Forest': RF_score
    }

    # Save the additive score to a file with the nutriscore and the name of the product
    with open('nutri_scores.csv', 'w') as f:
        header = 'nutriscore,G_0,PMS_0.5,PMS_0.6,PMS_0.7,OMS_0.5,OMS_0.6,OMS_0.7,Decision Tree,kNN,Random Forest\n'
        f.write(header)
        for i, row in dataset.iterrows():
            f.write(f'{row["nutriscore"]},{nutriscore[i]},{pms[0][i]},{pms[1][i]},{pms[2][i]},{oms[0][i]},{oms[1][i]},{oms[2][i]}, {decision_tree_nutriscore[i]}, {kNN_score[i]}, {RF_score[i]}\n')

    # Show the histogram of the scores
    show_scores_histogram(additive_score, bins=100)

    # Show the confusion matrix
    real_scores = [x if x in ['a', 'b', 'c', 'd', 'e'] else 'e' for x in dataset['nutriscore']]
    show_confusion_matrix(nutriscore, real_scores, 'Additive Nutriscore', 'Predicted', 'Real')
    for i in range(len(pms)):
        lambda_threshold = lambda_ex5[i]
        show_confusion_matrix(pms[i], real_scores, f'Pessimistic Majority Sorting (lambda={lambda_threshold})', 'Predicted', 'Real')
    for i in range(len(oms)):
        lambda_threshold = lambda_ex5[i]
        show_confusion_matrix(oms[i], real_scores, f'Optimistic Majority Sorting (lambda={lambda_threshold})', 'Predicted', 'Real')    
    show_confusion_matrix(decision_tree_nutriscore, real_scores, 'Decision Tree', 'Predicted', 'Real')
    show_confusion_matrix(kNN_score, real_scores, 'kNN', 'Predicted', 'Real')
    show_confusion_matrix(RF_score, real_scores, 'Random Forest', 'Predicted', 'Real')

    show_confusion_matrix(decision_tree_nutriscore, RF_score, 'Decision Tree vs Random Forest', 'Decision Tree', 'Random Forest')

    # Show the decision tree
    tree.plot_tree(clf, filled=True)
    plt.show()