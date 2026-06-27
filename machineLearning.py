"""
Stock market direction prediction from daily news headlines.

Predicts whether the Dow Jones Industrial Average (DJIA) rose or fell on a
given day using only that day's news headlines. Each day's headlines are
turned into a dense semantic embedding with a pretrained sentence transformer,
and a random forest classifier is trained on those embeddings.

Dataset: "Daily News for Stock Market Prediction" (Combined_News_DJIA.csv),
available on Kaggle. Place the CSV in the same directory as this file.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_FILE = 'Combined_News_DJIA.csv'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'


def get_data(connect=True):
    """Read the dataset and return (inputs, targets).

    Each row of the CSV is one day: column 1 is the label (1 if the DJIA rose,
    0 if it fell) and the remaining columns are that day's headlines.

    If connect is True, a day's headlines are joined into a single string;
    otherwise the day is kept as a list of separate headline strings.
    """
    with open(DATA_FILE) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # skip header row

        inputs = []
        targets = []
        for line in tqdm(list(csv_reader), desc='Reading CSV'):
            targets.append(int(line[1]))

            # Clean each headline: the raw data stores them as Python byte
            # strings, so strip the leading b'...' / b"..." markers and any
            # stray trailing quotes or backslashes.
            headlines = []
            for headline in line[2:]:
                headline = headline[:-1] if headline.endswith(('"', "'")) else headline
                headline = headline.replace("b'", '').replace('b"', '').replace('\\', '')
                headlines.append(headline)

            inputs.append(' '.join(headlines) if connect else headlines)

    return inputs, targets


def display_accuracy(targets, predictions, labels=('DOW fell', 'DOW rose'),
                     plot_title='Default title'):
    """Plot a labeled confusion matrix for a set of predictions."""
    cm = confusion_matrix(targets, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=list(labels))
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.show()


def run_model(limit_num_sentences=1001, connect=True, n_estimators=100):
    """Embed the headlines, train a random forest, and report accuracy.

    Computing the embeddings is the slow step, so they are cached to a .npz
    file keyed by the run parameters and reused on later runs. Tuning
    n_estimators and limit_num_sentences trades runtime for a more stable
    accuracy estimate.
    """
    save_file = Path(f'sentence_embeddings_{limit_num_sentences}_connect{connect}.npz')

    if save_file.exists():
        npz = np.load(save_file)
        inputs, targets = npz['inputs'], npz['targets']
        print(f'Loaded cached embeddings from {save_file}')
    else:
        model = SentenceTransformer(EMBEDDING_MODEL)
        sentences, targets = get_data(connect)
        sentences = sentences[:limit_num_sentences]
        targets = targets[:limit_num_sentences]

        embeddings = []
        for headline_row in tqdm(sentences, desc='Converting strings to embeddings'):
            embeddings.append(model.encode(headline_row))
        inputs = np.array(embeddings)
        targets = np.array(targets)

        np.savez(save_file, inputs=inputs, targets=targets)
        print(f'Saved embeddings to {save_file}')

    # When headlines are kept separate, collapse a day's per-headline
    # embeddings into one vector by taking the max across headlines.
    if not connect:
        inputs = inputs.max(axis=1)

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.10, random_state=0,
    )

    classifier = RandomForestClassifier(random_state=0, n_estimators=n_estimators)
    classifier.fit(inputs_train, targets_train)

    predictions_train = classifier.predict(inputs_train)
    predictions_test = classifier.predict(inputs_test)

    train_acc = (predictions_train == targets_train).mean() * 100
    test_acc = (predictions_test == targets_test).mean() * 100
    print(f'Train accuracy: {train_acc:.2f}%')
    print(f'Test accuracy:  {test_acc:.2f}%')

    display_accuracy(targets_train, predictions_train, plot_title='Train Performance')
    display_accuracy(targets_test, predictions_test, plot_title='Test Performance')


if __name__ == '__main__':
    run_model()
