from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError

from .base import Base
from .utils import get_or_create

class Experiment(Base):
    __tablename__ = 'experiment'

    id = Column(Integer, primary_key=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
    
    name = Column(String(255), nullable=False, unique=True) # Name of experiment, like "misspecified_class_average"

    # Either datasets or model_runs can refer to a parent experiment
    datasets = relationship("Dataset", back_populates='experiment', cascade="all, delete")
    model_runs = relationship("ModelRun", back_populates='experiment', cascade="all, delete")

    def __repr__(self):
        return f'(Experiment {self.name})'

def load_experiment_by_name(session, experiment_name):
    """
    Load an experiment by name, creating one if it doesn't already exist.
    """

    try:
        experiment = Experiment(name=experiment_name)
        session.add(experiment)
        session.commit()
    except IntegrityError:
        session.rollback()
        experiment = session.query(Experiment).filter(Experiment.name == experiment_name).one()
    return experiment
