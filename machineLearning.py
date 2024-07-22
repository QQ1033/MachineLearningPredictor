import csv

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


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
                headline = headline[2: -1]
                headline = headline.replace(r'\"', '"')
                headline = headline.replace(r"\'", "'")
                for func in feature_funcs:
                    inp_row.append(func(headline))
            # A few rows are short of having the necessary number of headlines;
            # append zeros if this row is short
            inp_row += (num_headlines - len(headlines)) * len(feature_funcs) * [0]
            inputs.append(inp_row)

    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def display_accuracy(targets, predictions, labels=['DOW fell', 'DOW rose'], plot_title='Default title'):
    cm = confusion_matrix(targets, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.show()

def main():
    inputs, targets = get_data()
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.10, random_state=0,
    )
    classifier = RandomForestClassifier(random_state=1, n_estimators=200, verbose=1)
    classifier.fit(inputs_train, targets_train)
    predictions_train = classifier.predict(inputs_train)
    predictions_test = classifier.predict(inputs_test)
    print(f'Train accuracy is {(predictions_train == targets_train).mean() * 100:.4f}%')
    print(f'Test accuracy is {(predictions_test == targets_test).mean() * 100:.4f}%')
    display_accuracy(targets_train, predictions_train, labels=['DOW fell', 'DOW rose'], plot_title='Train Performance')
    display_accuracy(targets_test, predictions_test, labels=['DOW fell', 'DOW rose'], plot_title='Test Performance')

if __name__ == '__main__':
    main()
