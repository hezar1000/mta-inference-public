"""
TODO: document inference loop
"""

import argparse
import csv
import glob
import json
import os
import pathlib
import pickle
import shutil
import subprocess
from tqdm import tqdm
from walrus import *
from collections import OrderedDict
from datetime import datetime
from explain_grades import optimize_explanation

import numpy as np

import logging

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

# TODO: set logging level in argparse like https://stackoverflow.com/a/20663028/3817091
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
log = logging.getLogger(__name__)
log.addHandler(TqdmLoggingHandler())

import tqdm

from load_data import load_mta_data, load_mta_data_from_redis
from inference_utils import convert_samples_to_bin_histograms

# Discrete grade bins for each inference scale
# Note that list of bins used for binning samples and reporting final grades must be same length 
grade_bin_lookup = {
    '5':  np.array([0, 1, 2, 3, 4, 5]),
    '25': np.array([0, 6.25, 12.5, 16.25, 20, 25]),
}

def loadConfig(config_path):
    """
    Load JSON config file as dictionary
    """
    logging.info('Loading config from %s' % config_path)
    with open(config_path, 'r') as f:
        return json.load(f)

# def runSamplers(input_dir, sample_dir, model, hyperparams, settings):
def runSamplers(input_dir, sample_dir, clamped_path, model, hyperparams, settings):

    """
    Run Gibbs sampling distributed across multiple processes.

    Inputs:
    - input_dir: folder containing submission CSVs. assumes that all files in this folder are submissions, 
      and that alphabetical order sorts them from first to last week.
    - sample_dir: folder to save sample files. saves samples in "0.pkl" through "(num_processes-1).pkl"
    - model: currently unused.
    - hyperparams: dictionary of {inference_hyperparam: value}
    - settings: dictionary of
        - num_processes: number of independent Gibbs sampling processes to run
        - num_samples: number of samples to take in each Gibbs sampling run
        - grade_scale: string indicating which grade bins to use
    """

    processes = []
    for i in range(settings['num_processes']):
        # Build inference command
        cmd = [
            'python3',
            'run_censored_inference.py',
            f'--input_dir {input_dir}',
            f'--output_path {sample_dir}/{i}.pkl',
            f'--clamped_path {clamped_path}',
            f'--excluded_reviews ' + ' '.join([str(review_id) for review_id in settings['excluded_reviews']]),
            f'--grade_scale {settings["grade_scale_inference"]}',
            f'--num_samples {settings["num_samples"]}',
            f'--seed {i}',
        ] + [
            f'--{hyperparam} {hyperparams[hyperparam]}' for hyperparam in hyperparams
        ]
        # Make one inference runner verbose to give a rough sense of progress
        if i == 0:
            cmd += ['--verbose']
        cmd_string = ' '.join(cmd)

        logging.info('Running %s' % cmd_string)
        p = subprocess.Popen(cmd_string, shell=True)
        processes.append(p)

    # Block until all processes finish
    for p in processes:
        p.wait()

def exportGrades(fname, week_num, final_grades, weights, graph, week_nums, graders, submissions, include_header=True):
    """
    Export grades file for a single week to be imported into MTA.
    Ignores calibrations and assignments from other weeks.

    Inputs:
    - fname: path to save grade file.
    - week_num: which week number to save into this file
    - final_grades: list of final grades for every assignment in course
    - weights: (num_graders, num_assignments) array with weight assigned to each grade
    - graph: (num_graders, num_assignments) binary array indicating graders for each assignment
    - week_nums: (num_assignments) list of week numbers for each assigmnent
    - graders: ordereddict of (student ID, student role)
    - submissions: ordereddict of (submission ID, submission type)
    - include_header: only write header on CSV if True 
    """

    # Build CSV header.
    # Assumes that there are at most 6 graders for a single assignment.
    # TODO: include "confidence" column?
    fieldnames = ['sub_id', 'sub_final_grade']
    for i in range(1, 20+1):
        fieldnames = fieldnames + [
            'review%d_id' % i, 
            'review%d_weight' % i, 
#             'review%d_grade' % i
        ]

    grader_list = list(graders.items())
    with open(fname, 'w', newline='') as f:
        # Set up CSV writer with optional header
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()

        # Output grades
        for i, (submission_id, submission_type) in enumerate(list(submissions.items())):
            # Skip calibrations and assignments from other weeks
            if week_nums[i] != week_num:
                continue

            if submission_type == 'calibration':
                continue
#                final_grade = final_grades[i]
#                final_grade_total = round(sum(final_grade), 2)
#                row_dict = {'sub_id': submission_id, 'sub_final_grade': final_grade}

            final_grade = final_grades[i]
            final_grade_total = round(sum(final_grade), 2)
            submission_graders = np.where(graph[:, i] == 1)[0]

            row_dict = {'sub_id': submission_id, 'sub_final_grade': final_grade_total}
            for (reviewer_num, grader_idx) in enumerate(submission_graders):            
                row_dict['review%d_id'     % (reviewer_num+1)] = grader_list[grader_idx][0]
                row_dict['review%d_weight' % (reviewer_num+1)] = np.round(weights[grader_idx, i], 4)
#                row_dict['review%d_grade'  % (reviewer_num+1)] = sum(observed_grades[grader_idx, i, :])

            writer.writerow(row_dict)

def exportcomponentGrades(fname, week_num, final_grades, week_nums, graders, submissions, include_header=True):

    # Build CSV header.
    # Assumes that there are at most 6 graders for a single assignment.
    # TODO: include "confidence" column?
    fieldnames = ['sub_id', 'sub_final_grade']
#     for i in range(1, 20+1):
#         fieldnames = fieldnames + [
#             'review%d_id' % i, 
#             'review%d_weight' % i, 
# #             'review%d_grade' % i
#         ]

    grader_list = list(graders.items())
    with open(fname, 'w', newline='') as f:
        # Set up CSV writer with optional header
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()

        # Output grades
        for i, (submission_id, submission_type) in enumerate(list(submissions.items())):
            # Skip calibrations and assignments from other weeks
            if week_nums[i] != week_num:
                continue

            # if submission_type == 'calibration':
            #     # continue
            #    final_grade = final_grades[i]
            #    final_grade_total = round(sum(final_grade), 2)
            #    row_dict = {'sub_id': submission_id, 'sub_final_grade': final_grade}

            final_grade = final_grades[i]
            # final_grade_total = round(sum(final_grade), 2)
            # submission_graders = np.where(graph[:, i] == 1)[0]

            row_dict = {'sub_id': submission_id, 'sub_final_grade': final_grade}
            # for (reviewer_num, grader_idx) in enumerate(submission_graders):            
            #     row_dict['review%d_id'     % (reviewer_num+1)] = grader_list[grader_idx][0]
            #     row_dict['review%d_weight' % (reviewer_num+1)] = np.round(weights[grader_idx, i], 4)
#                row_dict['review%d_grade'  % (reviewer_num+1)] = sum(observed_grades[grader_idx, i, :])

            writer.writerow(row_dict)

def exportbinmassgrades(fname, reported_grades, week_num, true_grade_histograms, weights, graph, week_nums, graders, submissions, include_header=True):
    fieldnames = ['sub_id', 'sub_final_grade', 'sub_components']
    for i in range(1, 20+1):
        fieldnames = fieldnames + [
            'review%d_id' % i, 
            'review%d_weight' % i, 
#             'review%d_grade' % i
        ]
    grader_list = list(graders.items())

    with open(fname, 'w', newline='') as f:
        # Set up CSV writer with optional header
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()

        for i, (submission_id, submission_type) in enumerate(list(submissions.items())):
            # Skip calibrations and assignments from other weeks
            if week_nums[i] != week_num:
                continue

            posterior_mass = np.transpose(true_grade_histograms[:,i,:])
            submission_graders = np.where(graph[:, i] == 1)[0]
            submission_reported_grades = reported_grades [np.where(graph[:, i] == 1), i]
            grader_weights = []
            for (reviewer_num, grader_idx) in enumerate(submission_graders):            
                grader_weights.append(np.round(weights[grader_idx, i]))

            weights = np.array[grader_weights]
            final_weights, final_grades, _, _ = optimize_explanation(
                submission_reported_grades, 
                posterior_mass, 
                weights / weights.sum(), 
                max_weight_change = 0.05,
                min_weight = 0.1,
                penalty_coeff = 1e-2
            )

            final_grade =  round(sum(true_grade_output_bins[final_grades]), 2)
            row_dict = {'sub_id': submission_id, 'sub_final_grade': final_grade, 'sub_components' : final_grades }

            for (reviewer_num, grader_idx) in enumerate(submission_graders):            
                row_dict['review%d_id'     % (reviewer_num+1)] = grader_list[grader_idx][0]
                row_dict['review%d_weight' % (reviewer_num+1)] = np.round(final_weights[grader_idx, i], 4)
#                row_dict['review%d_grade'  % (reviewer_num+1)] = sum(observed_grades[grader_idx, i, :])
            writer.writerow(row_dict)



def exportDependabilities(fname, dependabilities, dependability_lbs, graders, include_header=True):
    """
    Export CSV of dependabilities for each student to import into MTA.

    TODO: also have this code merge dependabilities with existing qualification info?
    
    Inputs:
    - fname: path to store dependabilities
    - dependabilities: list of mean dependability estimates
    - dependability_lbs: list of dependability lower bounds
    - graders: ordereddict of (student ID, 'student' or 'ta')
    - include_header: only write header on CSV if True 
    """
    fieldnames = ['Student ID', 'Lower Confidence Bound', 'Marking load', 'Upper Confidence Bound']
    with open(fname, 'w', newline='', ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        for i, (student_id, role) in enumerate(graders.items()):
            writer.writerow({
                'Student ID': student_id.strip(), 
                'Lower Confidence Bound': dependability_lbs[i],
                'Marking load': dependabilities[i], 
                'Upper Confidence Bound': 1 # unused
            })

def export_effort_and_reliabilities(fname, efforts, reliabilities, graders, include_header=True):
    """
    Export CSV of efforts and reliablities for each student to import into MTA.

    TODO: also have this code merge efforts and reliablities with existing qualification info?
    
    Inputs:
    - fname: path to store efforts and reliabilities
    - efforts : list of effort estimates
    -  reliabilities: list of reliability estimates
    - graders: ordereddict of (student ID, 'student' or 'ta')
    - include_header: only write header on CSV if True 
    """
    fieldnames = ['Student ID', 'Efforts', 'Reliabilities']
    with open(fname, 'w', newline='', ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        for i, (student_id, role) in enumerate(graders.items()):
            writer.writerow({
                'Student ID': student_id.strip(), 
                'Efforts': efforts[i],
                'Reliabilities': reliabilities[i]
            })

def summarizeSamples(input_dir, sample_dir, output_dir, lb_quantile, num_samples_discard, true_grade_sample_bins, true_grade_output_bins):
    # Load class data to align students and assignments
    # input_fnames = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    # _, graph, _, _, _, week_nums, graders, submissions, _ = load_mta_data(input_fnames)
    # week_nums = week_nums.astype(int)
    
    with open(input_dir,'rb') as f:
        reported_grades, graph, _, _, _, week_nums, graders, submissions, _ = pickle.load(f)
    
    week_nums = week_nums.astype(int)

    # Load samples
    true_grade_samples = []
    reliability_samples = []
    effort_samples = []
    effort_draw_samples = []

    # TODO: only load latest sample files?
    for fname in glob.glob(os.path.join(sample_dir, '*.pkl')):
        with open(fname, 'rb') as f:
            samples = pickle.load(f)
            true_grade_samples.append(samples['true_grades'][num_samples_discard:])
            reliability_samples.append(samples['reliabilities'][num_samples_discard:])
            effort_samples.append(samples['efforts'][num_samples_discard:])
            effort_draw_samples.append(samples['effort_draws'][num_samples_discard:])

    true_grade_samples = np.concatenate(true_grade_samples, axis=0)
    reliability_samples = np.concatenate(reliability_samples, axis=0)
    effort_samples = np.concatenate(effort_samples, axis=0)
    effort_draw_samples = np.concatenate(effort_draw_samples, axis=0)
    dependability_samples = effort_samples * reliability_samples
    efforts = effort_samples.mean(axis=0)

    # Aggregate samples
    true_grade_histograms = convert_samples_to_bin_histograms(true_grade_samples, true_grade_sample_bins)
    true_grades_likely = true_grade_histograms.argmax(axis=0)
    true_grades = true_grade_output_bins[true_grades_likely]
    # TODO: use probabilities in true_grade_histograms[true_grades_likely]?
    reliabilities = reliability_samples.mean(axis=0)
    effort_draws = effort_draw_samples.mean(axis=0)
    dependabilities = dependability_samples.mean(axis=0)
    dependability_lbs = np.quantile(dependability_samples, lb_quantile, axis=0)
    component_true_grades = true_grade_samples.mean(axis=0)
    # Summarize grades
    weights = reliabilities.reshape(-1, 1) * effort_draws
    for week_num in range(1, max(week_nums)+1):
        exportGrades(
            os.path.join(output_dir, 'grades-%02d.csv' % (week_num)),
            week_num,
            true_grades,
            weights,
            graph,
            week_nums,
            graders, 
            submissions, 
            include_header=True
        )
        exportcomponentGrades(
            os.path.join(output_dir, 'component-grades-%02d.csv' % (week_num)),
            week_num,
            component_true_grades,
            week_nums,
            graders, 
            submissions, 
            include_header=True
        )
        exportbinmassgrades(
            os.path.join(output_dir, 'bin-massees-%02d.pkl' % (week_num)),
            reported_grades,
            week_num,
            true_grade_histograms,
            weights,
            graph,
            week_nums,
            graders,
            submissions, 
            include_header=True
        )

    # Summarize dependabilities
    exportDependabilities(
        os.path.join(output_dir, 'dependabilities.csv'),
        dependabilities, 
        dependability_lbs, 
        graders,
        include_header=True,
    )
    # export effort and reliabilities
    export_effort_and_reliabilities(
        os.path.join(output_dir, 'efforts_and_reliabilities.csv'), 
        efforts, 
        reliabilities, 
        graders, 
        include_header=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference control loop.')
    parser.add_argument('--config', required=True, help='Path to JSON file containing config information')
    parser.add_argument('--run_once', action='store_true', help='Break after running control loop once')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference (for debugging)')
    parser.add_argument('--skip_zumbies_inference', action='store_true', help='Skip zumbies inference (for debugging)')

    logging.info('Starting inference runner')
    args = parser.parse_args()
    logging.info(args)

    # Main control loop
    while(True):
        # Load latest settings
        config = loadConfig(args.config)

        # Find directories
        input_db  = config['directories']['input_db']
        input_dir  = config['directories']['input']
        temp_dir  = config['directories']['input_working']
        sample_dir = config['directories']['samples']
        zumbies_sample_dir = config['directories']['zumbies_samples']
        output_dir = config['directories']['output']
        zumbies_output_dir = config['directories']['zumbies_output']
        logging.info('input:   %s' % input_db)
        logging.info('temp:    %s' % temp_dir)
        logging.info('samples: %s' % sample_dir)
        logging.info('output:  %s' % output_dir)

        #load data from redis 
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
            df \
        = load_mta_data_from_redis(input_db, excluded_submissions=excluded_submissions, verbose=True, passkey = passkey)

        zumbie_graph = np.copy(graph)
        zumbie_graph_temp = np.copy(graph)
        real_graph = np.copy(graph)
        zumbie_reported_grades = np.copy(reported_grades)
        zumbie_reported_grades_temp = np.copy(reported_grades)
        real_reported_grades = np.copy(reported_grades)
        zumbie_effort_grades = np.copy(effort_grades)
        zumbie_effort_grades_temp = np.copy(effort_grades)
        real_effort_grades = np.copy(effort_grades)

        window_size=config['inference_settings']['window_size']
        num_reviews_per_student = np.sum(graph,axis=1)
        for i,num_reviews in enumerate(num_reviews_per_student):
            if int(num_reviews) >= 2 * window_size:
                for j in range(int(num_reviews)-window_size):
                    index = np.where(ordered_graph[i]==j)
                    real_graph[i,index]=0
                    real_effort_grades[i,index]=0
                    real_reported_grades[i,index,:] = np.array([0,0,0,0])
                    
                for k in range(int(num_reviews)-window_size,int(num_reviews)):
                    index = np.where(ordered_graph[i]==k)
                    zumbie_graph[i,index]=0
                    zumbie_graph_temp[i,index]=0
                    zumbie_effort_grades[i,index]=0
                    zumbie_effort_grades_temp[i,index]=0
                    zumbie_reported_grades[i,index,:] = np.array([0,0,0,0])
                    zumbie_reported_grades_temp[i,index,:] = np.array([0,0,0,0])
            elif int(num_reviews) <  2 * window_size and int(num_reviews) > window_size:
                for j in range(int(num_reviews)-window_size):
                    index = np.where(ordered_graph[i]==j)
                    real_graph[i,index]=0
                    real_effort_grades[i,index]=0
                    real_reported_grades[i,index,:] = np.array([0,0,0,0])
                    
                for k in range(window_size,int(num_reviews)):
                    index = np.where(ordered_graph[i]==k)
                    zumbie_graph[i,index]=0
                    zumbie_effort_grades[i,index]=0
                    zumbie_reported_grades[i,index,:] = np.array([0,0,0,0])
   
                for k in range(int(num_reviews)-window_size,int(num_reviews)):
                    index = np.where(ordered_graph[i]==k)
                    zumbie_graph_temp[i,index]=0
                    zumbie_effort_grades_temp[i,index]=0
                    zumbie_reported_grades_temp[i,index,:] = np.array([0,0,0,0])
            else:
                for j in range(int(num_reviews)):
                    index = np.where(ordered_graph[i]==j)
                    zumbie_graph[i,index]=0
                    zumbie_graph_temp[i,index]=0
                    zumbie_effort_grades[i,index]=0
                    zumbie_effort_grades_temp[i,index]=0
                    zumbie_reported_grades[i,index,:] = np.array([0,0,0,0])
                    zumbie_reported_grades_temp[i,index,:] = np.array([0,0,0,0])
                    
        for i, (student, role) in enumerate(students.items()):
            if role == 'ta':
                zumbie_graph_temp[i,:]= 0
                real_graph[i,:]= graph[i,:]
                zumbie_graph[i,:]=0
                real_reported_grades[i,:,:] = reported_grades[i,:,:]
                real_effort_grades[i,:] = effort_grades[i,:]
                zumbie_effort_grades[i,:]=0
                zumbie_effort_grades_temp[i,:]=0
                zumbie_reported_grades[i,:,:] = 0
                zumbie_reported_grades_temp[i,:,:] = 0

        final_graph = np.concatenate((real_graph, zumbie_graph_temp), axis=0)
        final_effort_grades = np.concatenate((real_effort_grades, zumbie_effort_grades_temp), axis=0)
        final_reported_grades = np.concatenate((real_reported_grades, zumbie_reported_grades_temp), axis=0)
        
        # create new order_dict for students:
        zumbie_students = OrderedDict()
        for key in students:
            zumbie_students[key+'_zumbie'] = students[key]
        final_students = OrderedDict(list(students.items()) + list(zumbie_students.items()))
        # save as pickle files
        with open(input_dir+'/redis.pickle','wb') as f:
            # pickle.dump(load_mta_data_from_redis(input_db, excluded_submissions=[], verbose=True),f)
            pickle.dump([final_reported_grades,final_graph, ordered_graph, calibration_grades, final_effort_grades, week_nums, final_students, submissions, df],f)

        with open(input_dir+'/zumbie_redis.pickle','wb') as f:
            pickle.dump([zumbie_reported_grades,zumbie_graph, ordered_graph, calibration_grades, zumbie_effort_grades, week_nums, zumbie_students, submissions, df],f)
        
        today = datetime.now()
        sub_dir = today.strftime('%Y-%b-%d-%I-%M-%p')
        final_output_dir = output_dir + '/' + sub_dir
        final_sample_dir = sample_dir + '/' + sub_dir
        final_output_dir_zumbies = zumbies_output_dir + '/' + sub_dir
        final_sample_dir_zumbies = zumbies_sample_dir + '/' + sub_dir
        # Make output directory if it doesn't exist
        pathlib.Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(final_sample_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(final_output_dir_zumbies).mkdir(parents=True, exist_ok=True)
        pathlib.Path(final_sample_dir_zumbies).mkdir(parents=True, exist_ok=True)
        # Replace files in temp dir
       # logging.info('Copying inputs into temp dir...')
       # shutil.rmtree(temp_dir, ignore_errors=True)
       # shutil.copytree(input_dir, temp_dir)
        
        # Obtain Gibbs samples
        if args.skip_inference:
            logging.info('Skipping inference...')
        else:
            logging.info('Running Gibbs sampling...')
            
            # remove contents of sample folder before running samplers
        #    shutil.rmtree(sample_dir, ignore_errors=True)
        #    pathlib.Path(sample_dir).mkdir(parents=True, exist_ok=True)
            if args.skip_zumbies_inference:
                logging.info('Skipping zumbies inference...')
            else:
        #zumbies round:
                runSamplers(
                    # temp_dir, 
                    input_dir+'/zumbie_redis.pickle',
                    final_sample_dir_zumbies, 
                    'Not_needed', # clamped_path = 'Not needed'
                    config['inference_model'], 
                    config['inference_hyperparams'], 
                    config['inference_settings'],
                )
            
                # Summarize samples into true grades/dependabilities
                logging.info('Summarizing samples...')
            summarizeSamples(
                # temp_dir, 
                input_dir+'/zumbie_redis.pickle',
                final_sample_dir_zumbies, 
                final_output_dir_zumbies, 
                config['inference_settings']['lb_quantile'],
                config['inference_settings']['num_samples_discard'],
                grade_bin_lookup[config['inference_settings']['grade_scale_inference']],
                grade_bin_lookup[config['inference_settings']['grade_scale_output']],
            )
            reliabilities_prior=[]
            with open(final_output_dir_zumbies+'/efforts_and_reliabilities.csv', 'r', newline='', ) as f:
                rows= csv.reader(f, delimiter=',')
                for row in rows:
                    if not row[2]== 'Reliabilities':
                        reliabilities_prior.append(float(row[2]))
            hyperparams = config['inference_hyperparams']
            alpha_list =  ' '.join([str(alpha_tau) for alpha_tau in list(np.array(reliabilities_prior)**2)+list(np.array(reliabilities_prior)**2)+[1]])
            hyperparams['alpha_tau']= alpha_list
            beta_list =  ' '.join([str(beta_tau) for beta_tau in reliabilities_prior+reliabilities_prior+[1]])  
            hyperparams['beta_tau'] = beta_list
            # Real inference:
            runSamplers(
                # temp_dir, 
                input_dir+'/redis.pickle',
                final_sample_dir, 
                final_output_dir_zumbies,
                config['inference_model'],
                hyperparams, 
               # config['inference_hyperparams'], 
                config['inference_settings'],
            )
            
            # Summarize samples into true grades/dependabilities
        logging.info('Summarizing samples...')
        summarizeSamples(
            # temp_dir, 
            input_dir+'/redis.pickle',
            final_sample_dir, 
            final_output_dir, 
            config['inference_settings']['lb_quantile'],
            config['inference_settings']['num_samples_discard'],
            grade_bin_lookup[config['inference_settings']['grade_scale_inference']],
            grade_bin_lookup[config['inference_settings']['grade_scale_output']],
        )
        if args.run_once:
            break
