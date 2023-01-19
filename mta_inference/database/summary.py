from sqlalchemy import Column, Integer, LargeBinary, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base

from .compression import compress_data, dump_data

BLOB_LENGTH = (2**32)-1

class Summary(Base):
    __tablename__ = 'summary'

    id = Column(Integer, primary_key=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
    
    model_run_id = Column(Integer, ForeignKey('model_run.id', ondelete='CASCADE'), nullable=False)
    model_run = relationship("ModelRun", back_populates='summary')

    data = Column(LargeBinary(length=BLOB_LENGTH)) # Pickle containing results

    def __repr__(self):
        return f'Summary from run {self.model_run.name}'

def save_summary(session, model_run, summary_data, commit=True):
    data_dump = dump_data(summary_data)

    # get existing one if we have it
    try:
        summary = Summary(
            model_run = model_run,
            data = data_dump,
        )
        session.add(summary)
        session.commit()
    except: # TODO: which exception?
        session.rollback()
        summary = session.query(Summary).filter(Summary.model_run == model_run).one()
        summary.data = data_dump
        session.add(summary)
        session.commit()
    return summary

def load_summary(session, model_run):
    summary = session.query(Summary).filter(Summary.model_run_id == model_run.id).one()
    return summary
