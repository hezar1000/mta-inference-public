"""
Shared base class for all models.
See https://stackoverflow.com/questions/7478403/sqlalchemy-classes-across-files/7479122 for details...
"""

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
