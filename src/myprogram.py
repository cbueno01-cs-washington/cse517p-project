#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nltk
import ast
import json
from nltk import ngrams
from collections import defaultdict, Counter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, model = defaultdict(lambda: defaultdict(lambda: 0))):
        self.model = model
        
    @classmethod
    def load_training_data(cls):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        training_dir = os.path.join(parent_directory, 'traindata')
    
        data = []
        for file in os.listdir(training_dir):
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):
                with open(os.path.join(training_dir,filename), 'r', encoding='utf-8') as opened_file:
                    try:
                        content = opened_file.read()
                        # Process the content
                        data.extend(ngrams(content.lower(), 3))
                    except:
                        print("Could not open ", filename)

        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp[-2:])
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # Count bigrams
        for w1, w2, w3 in data:
            self.model[w1 + w2][w3] += 1

        # Transform to probabilities
        for w_seq in self.model:
            total_count = float(sum(self.model[w_seq].values()))
            for last_letter in self.model[w_seq]:
                self.model[w_seq][last_letter] /= total_count

    def run_pred(self, data):
        # your code here
        preds = []
        for inp in data:
            next_word_dict = self.model[inp.lower()]
            line_pred = []

            if next_word_dict:
                counter = Counter(next_word_dict)
                top3 = counter.most_common(3) 
                line_pred = [item[0] for item in top3]
            else:
                all_chars = string.ascii_letters
                line_pred = [random.choice(all_chars) for _ in range(3)]

            preds.append(''.join(line_pred))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            json.dump(self.model, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model = defaultdict(lambda: defaultdict(lambda: 0))
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            model.update(defaultdict(lambda: defaultdict(lambda: 0), json.load(f)))

        return MyModel(model)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
