from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base

class ModelRun(Base):
    __tablename__ = 'model_run'

    id = Column(Integer, primary_key=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

    name = Column(String(255), nullable=False) # Name of model run, like "BEM low effort"
    settings = Column(JSON) # Optional field for inference settings, model hyperparams, etc

    # Required dataset that this model was run on 
    dataset_id = Column(Integer, ForeignKey('dataset.id'), nullable=False)
    dataset = relationship("Dataset", back_populates='model_runs')
    
    # Optional experiment ID (if several experiments use the same dataset)
    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    experiment = relationship('Experiment', back_populates='model_runs', passive_deletes=True)

    # Children: samples/summary from model run
    samples = relationship('Samples', back_populates='model_run', passive_deletes=True)
    summary = relationship('Summary', back_populates='model_run', uselist=False, passive_deletes=True)


    __table_args__ = (UniqueConstraint('dataset_id', 'experiment_id', 'name', name='_unique_model_runs'),
                     )
    

    def __repr__(self):
        return f'(Model run {self.name} on dataset {self.dataset.name})'

def save_model_run(session, name, dataset, experiment=None, settings=None, commit=True):
    """
    Convenience function to create a model run.
    """

    model_run = ModelRun(
        name=name,
        dataset=dataset,
        experiment=experiment,
        settings=settings,
    )

    session.add(model_run)

    if commit:
        session.commit()

    return model_run

def load_model_run_by_name(session, model_run_name, dataset=None, experiment=None):
    """
    Load a single model run by name
    """ 
    model_run_query = session.query(ModelRun)\
        .filter(ModelRun.name == model_run_name)
    if dataset is not None:
        model_run_query = model_run_query.filter(ModelRun.dataset_id == dataset.id)
    if experiment is not None:
        model_run_query = model_run_query.filter(ModelRun.experiment_id == experiment.id)
    return model_run_query.one()
    

def load_model_runs(session, dataset):
    """
    Load all model runs for a dataset
    """ 
    model_runs = session.query(ModelRun).filter(ModelRun.dataset_id == dataset.id).all()
    return model_runs
    