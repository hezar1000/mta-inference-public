from sqlalchemy import Column, Integer, JSON, LargeBinary, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base
from .compression import compress_data

BLOB_LENGTH = (2**32)-1

class Samples(Base):
    __tablename__ = 'samples'

    id = Column(Integer, primary_key=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

    model_run_id = Column(Integer, ForeignKey('model_run.id', ondelete='CASCADE'), nullable=False)
    model_run = relationship("ModelRun", back_populates='samples')

    data = Column(LargeBinary(length=BLOB_LENGTH)) # Pickle containing samples, likelihoods, etc
    settings = Column(JSON) # Optional field for run settings (random seeds, etc)

    def __repr__(self):
        return '\n'.join([
            f'(Samples from run {self.model_run.name}',
            f'  - settings: {self.settings})'
        ])

def save_samples(session, model_run, sample_data, settings=None, commit=True):
    samples = Samples(
        model_run = model_run,
        data = compress_data(sample_data),
    )

    if settings is not None:
        samples.settings = settings

    session.add(samples)

    if commit:
        session.commit()

    return samples

def load_samples(session, model_run):
    all_samples = session.query(Samples).filter(Samples.model_run_id == model_run.id).all()
    return all_samples
