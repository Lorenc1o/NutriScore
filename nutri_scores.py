import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        if value >= 8:
            final.append('a')
        elif value >= 6 and value < 8:
            final.append('b')
        elif value >= 4 and value < 6:
            final.append('c')
        elif value >= 2 and value < 4:
            final.append('d')
        elif value < 2:
            final.append('e')
    
    return scores, final

def show_scores_histogram(scores, bins=10):
    # Show the histogram of the scores
    plt.hist(scores, bins=bins)
    plt.show()

def show_confusion_matrix(new_scores, initial_scores):
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
            
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Our score")
    ax.set_ylabel("Actual score")
    # Add a colorbar
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()



filename = 'data.csv'
columns = [#'energy', 
    'fat_g', 'sugar_g', 'sodium_mg', 'perc_fruit', 'fibers_g', 'proteins_g']
utility_functions = { #'energy': lambda x: 10-values_to_scale(filename, 'energy', [335,670,1005,1340,1675,2010,2345,2680,3015,3350]),
                        'fat_g': lambda x: 10-values_to_scale(filename, 'fat_g', [1,2,3,4,5,6,7,8,9,10]),
                        'sugar_g': lambda x: 10-values_to_scale(filename, 'sugar_g', [4.5,9,13.5,18,22.5,27,31,36,40,45]),
                        'sodium_mg': lambda x: 10-values_to_scale(filename, 'sodium_mg', [90,180,270,360,450,540,630,720,810,900]),
                        'perc_fruit': lambda x: 2*values_to_scale(filename, 'perc_fruit', [10,20,30,40,60]),
                        'fibers_g': lambda x: 2*values_to_scale(filename, 'fibers_g', [0.9,1.9,2.8,3.7,4.7]),
                        'proteins_g': lambda x: 2*values_to_scale(filename, 'proteins_g', [1.6,3.2,4.8,6.4,8])}

weights = [#1/6,
           1/6,1/6,1/6,1/6,1/6,1/6]

if __name__ == '__main__':
    # Calculate the additive score
    additive_score, nutriscore = additive_nutri_score(filename, columns, utility_functions, weights)

    print(len(additive_score), len(nutriscore))

    # Print the computed score and the nutriscore from the dataset for each product
    dataset = pd.read_csv(filename, sep=';')
    n_corrects = 0
    for i, row in dataset.iterrows():
        # Print the two scores and the comparison of their lowcase versions
        print(row['nutriscore'], nutriscore[i], additive_score[i], row['nutriscore'].lower() == nutriscore[i].lower())
        if row['nutriscore'].lower() == nutriscore[i].lower():
            n_corrects += 1

    # Print the percentage of correct scores
    print('Correct scores:', n_corrects / len(nutriscore))

    # Save the additive score to a file with the nutriscore and the name of the product
    with open('nutri_scores.csv', 'w') as f:
        f.write('nutriscore,additive_score,name\n')
        for i, row in dataset.iterrows():
            f.write(f'{row["nutriscore"]},{nutriscore[i]},{row["name"]}\n')

    print(max(additive_score), min(additive_score))

    # Show the histogram of the scores
    show_scores_histogram(additive_score, bins=100)

    # Show the confusion matrix
    real_scores = [x if x in ['a', 'b', 'c', 'd', 'e'] else 'e' for x in dataset['nutriscore']]
    show_confusion_matrix(nutriscore, real_scores)