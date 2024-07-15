import csv
import numpy as np

def func1(s):
    return 0

def func2(s):
    return 0

def func3(s):
    return 0

def main():
    with open('Combined_News_DJIA.csv') as csv_file:
        csv_reader = csv.reader(csv_file)

        # skip header but get num_headlines by ignoring first two columns
        num_headlines = len(next(csv_reader)) - 2
        inputs = []
        targets = []
        feature_funcs = [func1, func2, func3]

        for line in csv_reader:
            targets.append(int(line[1]))
            inp_row = []
            headlines = line[2:]
            for headline in headlines:
                for func in feature_funcs:
                    inp_row.append(func(headline))
            # append zeros if this row is short of having `num_headlines` headlines
            inp_row += (num_headlines - len(headlines)) * len(feature_funcs) * [0]
            inputs.append(inp_row)

        inputs = np.array(inputs)
        targets = np.array(targets)
        print(f'{inputs.shape = }')
        print(f'{targets.shape = }')

if __name__ == '__main__':
    main()
