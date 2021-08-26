#!/usr/bin/env python3
"""early stopping for reg"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """return bool if it should be stopped"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count != patience:
        return False, count
    else:
        return True, count
