"""
Run an entire inference experiment.

Example:
python control_inference_sockeye_experiments.py --database sqlite:////scratch/st-kevinlb-1/gregdeon/mta-inference/data/database.db --config /scratch/st-kevinlb-1/gregdeon/mta-inference/config/debug.json --dataset_name fall21 --run_name test_run --experiment_name debugging
"""

import argparse
import json
import os
import subprocess
from tqdm import tqdm
from explain_grades import optimize_explanation
from mta_inference.database.dataset import load_dataset
from mta_inference.run_censored_inference_sockeye import convertGradeScale
from mta_inference.experiment_utils import calculateLikelihoods, lpd, waic
from mta_inference.psis import psisloo
from scipy.stats import pearsonr , spearmanr


from mta_inference.database import create_session, load_experiment_by_name, load_model_run_by_name, save_model_run, save_summary, decompress_data

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

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def mae(x, y):
    return np.mean(np.abs(x - y))

samplers = {
    # # PG0/PG1/biases
    # 'censored_PG0_efforts_biases': 'run_PG5_inference --disable_correlation', 
    # 'censored_BEM': 'run_censored_inference_sockeye.py',
    # 'censored_PG1': 'run_censored_PG1_inference_sockeye.py',
    # 'censored_PG0': 'run_censored_inference_sockeye.py --disable_efforts',

    # # uncensored PG0/1
    # 'uncensored_PG0_efforts_biases': 'run_PG5_inference --disable_censoring --disable_correlation',
    # 'uncensored_BEM': 'run_component_inference_sockeye.py',
    # 'uncensored_PG1': 'run_PG1_inference_sockeye.py',
    # 'uncensored_PG0': 'run_PG5_inference --disable_censoring --disable_biases --disable_efforts --disable_correlation',
    # # uncensored PG1

    # PG0/PG1/biases
    'censored_PG_efforts_biases': 'run_PG5_inference.py --disable_correlation', 
    'censored_PG_efforts': 'run_censored_inference_sockeye.py',
    'censored_PG_biases': 'run_censored_PG1_inference_sockeye.py',
    'censored_PG': 'run_censored_inference_sockeye.py --disable_efforts',

    # uncensored PG0/1
    'uncensored_PG_efforts_biases': 'run_PG5_inference.py --disable_censoring --disable_correlation',
    'uncensored_PG_efforts': 'run_PG5_inference.py --disable_censoring --disable_biases --disable_correlation',
    'uncensored_PG_biases': 'run_PG1_inference_sockeye.py',
    'uncensored_PG': 'run_PG5_inference.py --disable_censoring --disable_biases --disable_efforts --disable_correlation',
    # uncensored PG1

    # legacy names
    'censored': 'run_censored_inference_sockeye.py', # censored BEM (PG0 + efforts)
    'component': 'run_component_inference_sockeye.py', # uncensored BEM (PG0 + efforts)
    'PG1': 'run_PG1_inference_sockeye.py', # uncensored PG1

    # PG5 + variants
    'censored_PG5_efforts_biases': 'run_PG5_inference.py',
    'censored_PG5_efforts': 'run_PG5_inference.py --disable_biases',
    'censored_PG5_biases': 'run_PG5_inference.py --disable_efforts',
    'censored_PG5': 'run_PG5_inference.py --disable_biases --disable_efforts',

    # uncensored PG5 + variants
    'uncensored_PG5_efforts_biases': 'run_PG5_inference.py --disable_censoring',
    'uncensored_PG5_efforts': 'run_PG5_inference.py --disable_censoring --disable_biases',
    'uncensored_PG5_biases': 'run_PG5_inference.py --disable_censoring --disable_efforts',
    'uncensored_PG5': 'run_PG5_inference.py --disable_censoring --disable_biases --disable_efforts',
}

# Allow hyperparams to be overridden on command line
# TODO: would be ideal to allow overwriting any arbitrary settings on the command line...
OVERRIDE_HYPERPARAMS = [
    'mu_s', 'sigma_s', # true grades
    'alpha_tau', 'beta_tau', # reliabilities
    'sigma_tau', 'lambda_tau', # PG5 reliabilities
    'mu_b', 'sigma_b', # biases
    'alpha_e', 'beta_e', 'tau_l', # efforts 
    'p_uniform_high_effort', 'p_uniform_low_effort', # uniform mixing
]

def runSamplers(database, model_run, model, hyperparams, settings):
    """
    Run Gibbs sampling distributed across multiple processes.

    Inputs:
    - database: path to database
    - model_run: model run to add samples to
    - model: name of model to use
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
            samplers[model],
            f'--database {database}',
            f'--model_run_id {model_run.id}',
            f'--excluded_reviews ' + ' '.join([str(review_id) for review_id in settings['excluded_reviews']]),
            f'--grade_scale {settings["grade_scale_inference"]}',
            f'--num_samples {settings["num_samples"]}',
            f'--seed {i}',
        ] + [
            f'--{hyperparam} {hyperparams[hyperparam]}' for hyperparam in hyperparams
        ]
        # if 'disable_efforts' in settings and settings['disable_efforts'] == True:
        #     cmd += ['--disable_efforts']
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

def computeMipGrades(reported_grades, true_grade_histograms, weights, graph):
    num_graders, num_assignments, num_components = reported_grades.shape
    mip_grades = np.full((num_assignments, num_components), np.nan)
    mip_weights = np.full((num_graders, num_assignments), 0)    
    mip_feasible = np.full((num_assignments), False)

    for i in tqdm(range(num_assignments)):
        # Find graders and their ideal weights
        posterior_mass = np.transpose(true_grade_histograms[:,i,:])
        submission_graders = np.where(graph[:, i] == 1)[0]
        submission_reported_grades = reported_grades[np.where(graph[:, i] == 1), i][0]
        submission_reported_grades = convertGradeScale(submission_reported_grades)
        grader_weights = np.round(weights[submission_graders, i], 2)

        try:
            # Optimize weighted sum
            # add small constant to weights to avoid issues with 0 weight students
            grader_weights_normalized = (grader_weights + 1e-3) / (grader_weights + 1e-3).sum()

            final_weights, final_grades, _, _ = optimize_explanation(
                submission_reported_grades, 
                posterior_mass, 
                grader_weights_normalized, 
                max_weight_change = 0.09,
                min_weight = 0.1,
                penalty_coeff = 1e-2
            )

            mip_grades[i, :] = final_grades
            mip_weights[submission_graders, i] = final_weights
            mip_feasible[i] = True

        except ValueError:
            # MIP was infeasible; nothing else to do
            logging.warning(f'MIP {i} infeasible! Inputs:')
            logging.warning(submission_reported_grades)
            logging.warning(posterior_mass)
            logging.warning(grader_weights_normalized)
            continue

    return mip_grades, mip_weights, mip_feasible

def summarizeSamples(model_run, hyperparams, num_samples_discard, true_grade_sample_bins):
    """
    Output results.
    """

    results_dict = {}

    # Get dataset
    reported_grades, graph, _, _, _, _, _, _ = decompress_data(model_run.dataset.data)

    # Load samples, discarding burn-in samples
    logging.info('Loading sample files...')
    true_grade_samples = []
    reliability_samples = []
    effort_samples = []
    effort_draw_samples = []
    bias_samples = []

    for samples_db in model_run.samples:
        samples = decompress_data(samples_db.data)
        true_grade_samples.append(samples['true_grades'][num_samples_discard:].astype(np.float64))
        reliability_samples.append(samples['reliabilities'][num_samples_discard:])
        if 'efforts' in samples:
            effort_samples.append(samples['efforts'][num_samples_discard:])
            effort_draw_samples.append(samples['effort_draws'][num_samples_discard:])
        if 'biases' in samples:
            bias_samples.append(samples['biases'][num_samples_discard:])
    
    has_efforts = (len(effort_samples) > 0)
    has_biases = (len(bias_samples) > 0)

    # Concatenate samples
    true_grade_samples = np.concatenate(true_grade_samples, axis=0)
    reliability_samples = np.concatenate(reliability_samples, axis=0)
    if has_efforts:
        effort_samples = np.concatenate(effort_samples, axis=0)
        effort_draw_samples = np.concatenate(effort_draw_samples, axis=0)
        dependability_samples = effort_samples * reliability_samples
    if has_biases:
        bias_samples = np.concatenate(bias_samples, axis=0)

    # Calculate mean estimates
    logging.info('Calculating sample means...')
    true_grade_means = true_grade_samples.mean(axis=0)
    reliability_means = reliability_samples.mean(axis=0)
    reliability_lower_cis = np.quantile(reliability_samples, 0.025, axis=0)
    reliability_higher_cis = np.quantile(reliability_samples, 0.975, axis=0)
    results_dict.update({
        'true_grades_mean': true_grade_means,
        'reliabilities': reliability_means,
        'reliability_lower_cis': reliability_lower_cis, 
        'reliability_higher_cis': reliability_higher_cis
    })
    if has_efforts:
        effort_means = effort_samples.mean(axis=0)
        effort_draw_means = effort_draw_samples.mean(axis=0)
        dependability_means = dependability_samples.mean(axis=0)
        results_dict.update({
            'efforts': effort_means,
            # 'effort_draws': effort_draw_means, # TODO: add option to save mean effort draws?
            'dependabilities': dependability_means,
        })
    if has_biases:
        bias_means = bias_samples.mean(axis=0)
        results_dict.update({
            'biases': bias_means
        })

    # Calculate discrete grade estimates
    logging.info('Building true grade posteriors...')
    true_grade_histograms = convert_samples_to_bin_histograms(true_grade_samples, true_grade_sample_bins)
    true_grades_likely = true_grade_histograms.argmax(axis=0)
    if has_efforts:
        instructor_graph = samples['instructor_graph']
        effort_draw_means_expanded = np.zeros_like(instructor_graph)
        effort_draw_means_expanded[np.where(instructor_graph)] = effort_draw_means
        inference_weights = reliability_means.reshape(-1, 1)[:-1] * effort_draw_means_expanded[:-1]
    else:
        # remove instructor
        inference_weights = reliability_means.reshape(-1, 1)[:-1] * graph
    # logging.info('Calculating true grades with MIP...')
    # true_grades_mip, mip_weights, mip_feasible = computeMipGrades(reported_grades, true_grade_histograms, inference_weights, graph)
    # results_dict.update({
    #     'true_grades_most_likely': true_grades_likely,
    #     'true_grades_mip': true_grades_mip,
    #     # TODO: add flag to save large outputs
    #     # 'inference_weights': inference_weights,
    #     # 'mip_weights': mip_weights,
    #     'mip_feasible': mip_feasible,
    # })

    # Calculate likelihoods
    logging.info('Computing likelihoods...')
    for (data_compressed, data_name) in [
        # (model_run.dataset.data, 'in_sample'), # TODO: add option to enable in-sample likelihoods?
        (model_run.dataset.held_out_data, 'held_out')
    ]:
        if data_compressed is not None:
            logging.info(f'- {data_name}...')
            reported_grades_ll, graph_ll, _, _, _, _, _, _ = decompress_data(data_compressed)

            likelihoods_mean = calculateLikelihoods(
                graph=graph_ll, reported_grades=reported_grades_ll, true_grades=true_grade_means, reliabilities=reliability_means[:-1],
                uniform_prob_high_effort=hyperparams['p_uniform_high_effort'], uniform_prob_low_effort=hyperparams['p_uniform_low_effort'], 
                efforts=effort_means[:-1] if has_efforts else None, 
                biases=bias_means[:-1] if has_biases else None, 
                mu_s=hyperparams['mu_s'],
                tau_l=hyperparams['tau_l'] if has_efforts else None,
            )

            num_samples = len(true_grade_samples)
            num_reports = len(likelihoods_mean)
            # Save per-sample likelihoods as float16 to cut storage space by 4
            likelihoods_samples = np.zeros((num_samples, num_reports))
            for sample in tqdm(range(num_samples)):
                likelihoods_samples[sample, :] = calculateLikelihoods(
                    graph=graph_ll, reported_grades=reported_grades_ll, true_grades=true_grade_samples[sample], reliabilities=reliability_samples[sample][:-1],
                    uniform_prob_high_effort=hyperparams['p_uniform_high_effort'], uniform_prob_low_effort=hyperparams['p_uniform_low_effort'], 
                    efforts=effort_samples[sample][:-1] if has_efforts else None, 
                    biases=bias_samples[sample][:-1] if has_biases else None, 
                    mu_s=hyperparams['mu_s'],
                    tau_l=hyperparams['tau_l'] if has_efforts else None,
                )

            observation_lpds = lpd(likelihoods_samples)
            _, _, observation_elpds, observation_p_waics = waic(likelihoods_samples)
            _, observation_loos, loo_ks = psisloo(likelihoods_samples)

            results_dict.update({
                f'log_likelihoods_{data_name}_mean': likelihoods_mean,
                f'log_likelihoods_{data_name}_lpd': observation_lpds,       # individual grades log likelihoods acrosss gibbs samples
                f'log_likelihoods_{data_name}_waic': observation_elpds,     # WAIC estimate of out-of-sample log likelihoods
                f'log_likelihoods_{data_name}_p_waic': observation_p_waics, # WAIC corrections
                f'log_likelihoods_{data_name}_loo': observation_loos,       # PSIS LOO estimate of out-of-sample log likelihoods
                f'log_likelihoods_{data_name}_loo_k': loo_ks,               # PSIS LOO k values (> 0.7 is potential issue)
                # TODO: add flag to save large outputs
                # f'log_likelihoods_{data_name}_samples': likelihoods_samples.astype(dtype=np.float16), # per-sample likelihoods
            })

    # Compute accuracy metrics if true_values are present
    logging.info('Computing accuracy metrics...')
    if model_run.dataset.true_values is not None:
        true_values = decompress_data(model_run.dataset.true_values)
        
        # TODO: add accuracy metrics from Sockeye code
        logging.warning('Accuracy metrics not implemented')


    return results_dict

def run_experiment_without_zombies(args, config):
    with create_session(args.database) as session:
        dataset = load_dataset(session, args.dataset_name)
        if args.experiment_name is not None:
            experiment = load_experiment_by_name(session, args.experiment_name)
        else:
            experiment = None
            
        if args.skip_inference:
            logging.info('Skipping inference and loading existing samples...')
            model_run = load_model_run_by_name(session, args.run_name, dataset, experiment)

        else:
            logging.info(f'Running inference on {args.dataset_name}...')

            model_run_args = vars(args).copy()
            del model_run_args['database']
            model_run = save_model_run(session, args.run_name, dataset, experiment, {'args': model_run_args, 'config': config}, commit=True)
            
            # Run Gibbs sampling
            runSamplers(
                args.database,
                model_run,
                config['inference_model'], 
                config['inference_hyperparams'], 
                config['inference_settings'],
            )

        
    # Summarize samples into true grades/dependabilities
    logging.info('Summarizing samples...')
    with create_session(args.database) as session:
        dataset = load_dataset(session, args.dataset_name)
        if args.experiment_name is not None:
            experiment = load_experiment_by_name(session, args.experiment_name)
        else:
            experiment = None

        model_run = load_model_run_by_name(session, args.run_name, dataset, experiment)
        results = summarizeSamples(
            model_run,
            config['inference_hyperparams'],
            config['inference_settings']['num_samples_discard'],
            grade_bin_lookup[config['inference_settings']['grade_scale_inference']],
        )

        save_summary(session, model_run, results, commit=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference control loop.')
    parser.add_argument('--config', required=True, help='Path to JSON file containing config information')
    parser.add_argument('--skip_inference', action='store_true', help='Skip all inference (for debugging)')
    parser.add_argument('--skip_zombies', action='store_true', help='Skip zombie inference and summarization (for debugging)')
    parser.add_argument('--skip_zombie_inference', action='store_true', help='Skip zombie inference (for debugging)')

    parser.add_argument('--database', type=str, required=True, help='Path to database')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of dataset to run inference on')
    parser.add_argument('--experiment_name', type=str, help='Optional name of model run experiment')
    parser.add_argument('--run_name', type=str, required=True, help='Name of model run')

    # Add overrideable hyperparams
    for hyperparam in OVERRIDE_HYPERPARAMS:
        parser.add_argument(f'--{hyperparam}', type=float)
    
    args = parser.parse_args()
    logging.info(args)
 
    # Load inference config
    config = loadConfig(args.config)

    # Set defaults for new settings to keep backward compatibility
    config['inference_settings'].setdefault('use_zombies', True)
    config['inference_hyperparams'].setdefault('p_uniform_high_effort', 0.05)
    config['inference_hyperparams'].setdefault('p_uniform_low_effort', 0.05)

    # Overwrite inference hyperparams from command line 
    for hyperparam in OVERRIDE_HYPERPARAMS:
        arg = getattr(args, hyperparam)
        if arg is not None:
            config['inference_hyperparams'][hyperparam] = arg

    run_experiment_without_zombies(args, config)
    