__author__ = 'Sartaaj Khan'


from warnings import warn
from .core import *
import sys


if sys.version_info.major < 3:
    warn(Exception('Require Python >= 3.5'))
