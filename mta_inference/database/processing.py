"""
Functions for post-processing results from the database.
"""

import pandas as pd
from tqdm import tqdm

#from mta_inference.run_PG5_inference import HYPERPARAMS
from . import *

def allbut(df, columns=[]):
    """
    Return a list of all columns in a dataframe, except for the specified ones.
    """
    return [c for c in df.columns if c not in columns]

def load_cross_validation_results(session, experiment_name):
    """
    Load total log likelihoods for all model runs in an experiment.
    
    - `session`: SQLAlchemy session
    - `experiment_name`: Name of experiment to load results for.
    """

    experiment = session.query(Experiment).filter_by(name=experiment_name).one()
    model_runs = session.query(ModelRun).filter(ModelRun.experiment_id == experiment.id).all()

    records = []
    for model_run in tqdm(model_runs):
        # Ignore runs that haven't finished
        if model_run.summary is None:
            continue

        # Override hyperparameters with run-specific hyperparameters
        # TODO: pretty sure this is unnecessary, since this overriding already happens before the run is started
        model_run_settings = model_run.settings
        hyperparams = model_run_settings['config']['inference_hyperparams']
        for arg in model_run_settings['args']:
            if arg in hyperparams and model_run_settings['args'][arg] is not None:
                hyperparams[arg] = model_run_settings['args'][arg]
        
        # Split dataset name into base name and fold
        # TODO: Save fold in dataset settings; don't parse from name
        dataset_name = model_run.dataset.name
        dataset_name_split = dataset_name.split('_')
        dataset_name_base = '_'.join(dataset_name_split[:-1])
        dataset_fold = int(dataset_name_split[-1])

        # Get held out likelihood
        summary_data = undump_data(model_run.summary.data)
        held_out_ll = summary_data['log_likelihoods_held_out_lpd'].sum()

        # Build record
        records.append({
            'model_run': '_'.join(model_run.name.split('_')[:-1]),
            'dataset': dataset_name_base,
            'fold': dataset_fold,
            **hyperparams,
            'held_out_ll': held_out_ll,
        })

    df = pd.DataFrame.from_records(records).fillna(0)
    df_grouped = df.groupby(allbut(df, ['fold', 'held_out_ll']))
    df_summed = df_grouped.apply(lambda group: pd.Series([group['held_out_ll'].sum(), int(group.shape[0])], index=['held_out_ll', 'num_folds']))
    return df_summed.reset_index(), df
