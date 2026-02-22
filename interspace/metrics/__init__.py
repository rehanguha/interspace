# -*- coding: utf-8 -*-
"""
Metrics module.

This module contains utility functions for working with distances.
"""

from interspace.metrics.pairwise import pairwise_distance
from interspace.metrics.validation import is_distance_metric

__all__ = ["pairwise_distance", "is_distance_metric"]