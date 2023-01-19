"""
Export anonymized class data to be used on other compute clusters. 

Example usage:
python export_data.py --output_file ../data/fall21_anonymized_with_authors.pbz2
"""

import sys
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import bz2
import pickle
import glob
import os

from mta_inference.load_data import load_mta_data, load_mta_data_from_redis
from mta_inference.inference_utils import convertGradeScale25To5

def loadConfig(config_path):
    """
    Load JSON config file as dictionary
    """
    logging.info('Loading config from %s' % config_path)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export class data for use on Sockeye.')
    parser.add_argument('--data_format', type=str, choices=['redis', 'csv'], default='redis')
    parser.add_argument('--config_file', type=str, default='../config/fall21.json') # used for Redis mode
    parser.add_argument('--csv_path', type=str) # used for csv mode
    parser.add_argument('--output_file', type=str, default='../export/fall21_anonymized.pbz2')
    
    args = parser.parse_args()
    print(args)

    if args.data_format == 'redis':
        # load config file
        with open(args.config_file, 'r') as f:
            config = json.load(f)

        # load data from redis
        logging.info('Loading data from Redis...')
        input_db  = config['directories']['input_db']
        excluded_submissions = config['inference_settings']['excluded_reviews']
        passkey = config['redis_passkey']
        reported_grades, \
            graph, \
            ordered_graph, \
            calibration_grades, \
            effort_grades, \
            week_nums, \
            students, \
            submissions, \
            _ \
        = load_mta_data_from_redis(input_db, excluded_submissions=excluded_submissions, verbose=True, passkey = passkey)
    elif args.data_format == 'csv':
        csv_files = sorted(glob.glob(os.path.join(args.csv_path, '*.csv')))
        reported_grades, \
            graph, \
            ordered_graph, \
            calibration_grades, \
            effort_grades, \
            week_nums, \
            students, \
            submissions, \
            _ \
        = load_mta_data(csv_files, excluded_submissions=[], verbose=True)
        
    else:
        raise ValueError(f'Unrecogized data format: {args.data_format}')

    # convert grades to 0-5 scale
    reported_grades_rescaled = convertGradeScale25To5(reported_grades)
    calibration_grades_rescaled = convertGradeScale25To5(calibration_grades)

    # anonymize
    logging.info('Anonymizing students...')
    students_anonymized = OrderedDict()
    for i, student_id in enumerate(students):
        students_anonymized[str(i+1)] = students[student_id]
        for sub_id, sub_info in submissions.items():
            if int(student_id) == int(sub_info[1]):
                submissions[sub_id]= [sub_info[0], str(i+1)]
        

    # save
    logging.info(f'Exporting results to {args.output_file}...')
    Path(args.output_file).parents[0].mkdir(parents=True, exist_ok=True)
    with bz2.open(args.output_file, 'wb') as f:
        pickle.dump([
            reported_grades_rescaled,
            graph,
            ordered_graph,
            calibration_grades,
            effort_grades,
            week_nums,
            students_anonymized,
            submissions,
        ], f)
