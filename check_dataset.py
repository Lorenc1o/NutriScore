import pandas as pd

def load_dataset(dataset_location, sep=';'):
    # Load the dataset
    dataset = pd.read_csv(dataset_location, sep=sep)
    
    # Return the dataset
    return dataset

def check_dataset(dataset1_location, dataset2_location, field1, field2, sep1=';', sep2=';'):
    # Load the datasets
    dataset1 = load_dataset(dataset1_location, sep=sep1)
    dataset2 = load_dataset(dataset2_location, sep=sep2)
    
    # Get the values of the specified field from each dataset
    dataset1_values = dataset1[field1]
    dataset2_values = dataset2[field2]
    
    # Calculate the percentage of coinciding values
    coinciding_percentage = len(set(dataset1_values) & set(dataset2_values)) / len(dataset1_values)
    
    # Check if less than 20% of the values coincide
    if coinciding_percentage < 0.2:
        return True
    else:
        return False
    
if __name__ == '__main__':
    # Check if the datasets are different
    different = check_dataset('data.csv', 'data1.csv', 'id', 'id', sep2=',')
    
    # Print the result
    print('The datasets are different:', different)

    # And the other way around
    different = check_dataset('data1.csv', 'data.csv', 'id', 'id', sep1=',')
    print('The datasets are different:', different)

    print('-' * 80)

    different2 = check_dataset('data.csv', 'data2.csv', 'id', 'code', sep2=',')
    print('The datasets are different:', different2)

    different2 = check_dataset('data2.csv', 'data.csv', 'code', 'id', sep1=',')
    print('The datasets are different:', different2)

    print('-' * 80)


    different3 = check_dataset('data.csv', 'data3.csv', 'id', 'code', sep2=',')

    print('The datasets are different:', different3)

    different3 = check_dataset('data3.csv', 'data.csv', 'code', 'id', sep1=',')
    print('The datasets are different:', different3)