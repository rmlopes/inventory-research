#   InventoryResearch - A library for the study of inventory management
#   Copyright (C) 2015-2016 Rui L. Lopes
#
#   This file is part of InventoryResearch.
#
#   InventoryResearch is free software: you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   InventoryResearch is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with InventoryResearch.  If not, see <http://www.gnu.org/licenses/>.
'''
Description
===========

A Python library for the study of inventory management. Implements cost functions, heuristics, and datasets used in Inventory Management research.

Using any of the following subpackages requires an explicit import.  For example,
``import inventory.discrete``.

::

 continuous                 --- Cost functions for the continuous model
 discrete                   --- Cost functions for the discrete model
 solvers                    --- Heuristics and optimal algorithms
 utils                      --- Utilities for the datasets and GP

'''

__author__ = "Rui L. Lopes"
__version__ = "0.1"
__revision__ = "0.1a"

import logging
#rootlvl = logging.getLogger('root').getEffectiveLevel()
logging.getLogger('inventory').addHandler(logging.NullHandler())
_FORMAT = '%(name)s: %(levelname)s:  %(message)s'
logging.basicConfig(format=_FORMAT, level = logging.INFO)
