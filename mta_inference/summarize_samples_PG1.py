"""
Calculate means and quantiles from Gibbs samples

Data structure is: 

- results['true_grades']['mean' or quantile][assignment][component]
with some grader/assignment/component axes missing
exception:
"""

import argparse
import glob
import pickle
import bz2
import _pickle as cPickle
import numpy as np
from collections import defaultdict

# TODO: configurable quantile list?
quantile_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize Gibbs samples with means and quantiles.')
    parser.add_argument('input_paths', help='List of paths with samples', nargs=argparse.ONE_OR_MORE)
    parser.add_argument('--output_path', required=True, help='Location to save summary')
    parser.add_argument('--drop_first', type=int, default=0, help='For burn-in, drop the first (drop_first) samples.')

    args = parser.parse_args()

    # Load data
    # TODO: do this without loading entire dataset into memory? only load one variable at a time?
    print('Loading data...')
    samples_stacked = defaultdict(list)
    for path in args.input_paths:
        for fname in glob.glob(path):
            print('- %s' % fname)
            samples_loaded = bz2.BZ2File(fname, 'rb')
            samples_loaded = cPickle.load(samples_loaded)
            for var in samples_loaded:
                samples_stacked[var].append(samples_loaded[var][args.drop_first:])

    # Compute means and quantiles for most variables
    print('Computing statistics...')
    summary = defaultdict(dict)
    for var in ['true_grades']:
        print('- %s' % var)
        samples = np.vstack(samples_stacked[var])
        summary[var]['mean'] = np.average(samples, axis=0)
        for quantile in quantile_list:
            summary[var][quantile] = np.quantile(samples, quantile, axis=0)


    print('Writing to %s...' % args.output_path)
    with bz2.BZ2File(args.output_path+fname[-42:], 'wb') as f:
        cPickle.dump(summary, f)
