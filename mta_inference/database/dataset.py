from sqlalchemy import Column, Integer, String, JSON, LargeBinary, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base
from .utils import get_or_create
from .experiment import Experiment
from .compression import compress_data

BLOB_LENGTH = (2**32)-1

class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

    name = Column(String(255), nullable=False, unique=True)
    data = Column(LargeBinary(length=BLOB_LENGTH), nullable=False)
    held_out_data = Column(LargeBinary(length=BLOB_LENGTH))
    true_values = Column(LargeBinary(length=BLOB_LENGTH))
    settings = Column(JSON)

    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    experiment = relationship("Experiment", back_populates='datasets')
    model_runs = relationship("ModelRun", back_populates='dataset', cascade="all, delete")

    def __repr__(self):
        return f'(Dataset {self.name})'

def save_dataset(session, name, data, held_out_data=None, true_values=None, settings=None, experiment_name=None, commit=True):
    dataset = Dataset(
        name=name,
        data=compress_data(data),
        held_out_data=compress_data(held_out_data) if held_out_data is not None else None,
        true_values=compress_data(true_values) if true_values is not None else None,
        settings=settings,
        experiment=get_or_create(session, Experiment, name=experiment_name) if experiment_name is not None else None,
    )

    session.add(dataset)

    if commit:
        session.commit()

    return dataset

def load_dataset(session, name):
    return session.query(Dataset).filter(Dataset.name == name).one()
