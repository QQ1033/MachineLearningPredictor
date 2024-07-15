import csv

import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

def func1(s):
    return len(s.split()) / (len(s) - s.count(' '))

def func2(s):
    count = 0
    for char in s:
        if (s.upper()):
            count += 1

    return count

def func3(s):
    return int('%' in s)

def func4(s):
    s = s.split()
    if 'Dow Jones' in s or 'DJIA' in s:
        return 10.0

    return 0.0

def func5(s):
    s = s.split()
    if 'Conflict' in s or 'War' in s:
        return -5.0

    return 0.0

def func6(s):
    count = 0
    for char in s:
        if char.isdigit():
            count += 1

    return count / (len(s) - s.count(' '))

def func7(s):
    count = 0
    for word in s.split():
        word = word.lower()
        if ('politics' in word or 'politician' in word
                or 'leader' in word or 'president' in word):
            count += 1

    return count

def func8(s):
    return int('?' in s)

def func9(s):
    count = 0
    for word in s.split():
        word = word.lower()
        if ('and' in word or 'or' in word
                or 'the' in word or 'for' in word or 'in' in word):
            count += 1

    return count

def func10(s):
    return len(s)

def get_data():
    with open('Combined_News_DJIA.csv') as csv_file:
        csv_reader = csv.reader(csv_file)

        # skip header but get num_headlines by ignoring first two columns
        num_headlines = len(next(csv_reader)) - 2
        inputs = []
        targets = []
        feature_funcs = [
            func1, func2, func3, func4, func5,
            func6, func7, func8, func9, func10
        ]

        for line in tqdm(list(csv_reader)):
            targets.append(int(line[1]))
            inp_row = []
            headlines = line[2:]
            for headline in headlines:
                for func in feature_funcs:
                    inp_row.append(func(headline))
            # A few rows are short of having the necessary number of headlines;
            # append zeros if this row is short
            inp_row += (num_headlines - len(headlines)) * len(feature_funcs) * [0]
            inputs.append(inp_row)

    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def main():
    inputs, targets = get_data()
    classifier = MLPClassifier(random_state=0, verbose=1)
    test_size = 10
    classifier.fit(inputs[test_size:], targets[test_size:])
    predictions = classifier.predict(inputs[:test_size])
    print(f'{predictions = }')
    print(f'{targets[:test_size] = }')

if __name__ == '__main__':
    main()
