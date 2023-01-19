from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from .base import Base

import logging
import sys
logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


engines = {}
def create_or_get_engine(engine_string, verbose=True):
    if engine_string in engines:
        engine = engines[engine_string]
        if verbose:
            logging.info(f'Loaded existing engine {repr(engine.url)}')
            
    if engine_string not in engines:
        engine = create_engine(engine_string, pool_recycle=3600) # reconnect once an hour
        Base.metadata.create_all(engine)
        engines[engine_string] = engine
        logging.info(f'Created new engine {repr(engine.url)}')

    return engine

def create_session(engine_string='sqlite:///example.db', verbose=True):
    """
    Create a session to connect to the database 
    """    

    engine = create_or_get_engine(engine_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def get_or_create(session, model, create_method='', create_method_kwargs=None, **kwargs):
    """
    adapted from https://stackoverflow.com/a/21146492/3817091
    """
    try:
        return session.query(model).filter_by(**kwargs).one(), False
    except NoResultFound:
        kwargs.update(create_method_kwargs or {})
        created = getattr(model, create_method, model)(**kwargs)
        try:
            session.add(created)
            session.flush()
            return created, True
        except IntegrityError:
            session.rollback()
            return session.query(model).filter_by(**kwargs).one(), False

