# -*- coding: utf-8 -*-
"""
String distance functions.

This module contains distance functions for comparing strings and sequences.
"""

from __future__ import annotations

import numpy as np


def hamming(a, b):
    """Hamming distance for ints (bitwise), strings/iterables (per-position mismatch).

    The Hamming distance measures the minimum number of substitutions required
    to change one string into the other, or the number of positions where
    the corresponding symbols are different.

    Parameters
    ----------
    a : int, str, or array_like
        First input.
    b : int, str, or array_like
        Second input.

    Returns
    -------
    int or float
        For integers: count of differing bits.
        For sequences: number of positions with differing elements.

    Raises
    ------
    ValueError
        If sequences have unequal length.

    Examples
    --------
    >>> hamming(0b1010, 0b0011)
    2
    >>> hamming("abcd", "abcf")
    1
    >>> hamming([1, 2, 3], [1, 0, 3])
    1

    Notes
    -----
    For integers, this uses XOR and popcount to count differing bits.
    For strings and arrays, elements must have the same length.
    """
    # integers: bitwise xor popcount
    if isinstance(a, int) and isinstance(b, int):
        return bin(a ^ b).count("1")

    # strings or sequence-like: compare lengths then elementwise
    if isinstance(a, str) and isinstance(b, str):
        if len(a) != len(b):
            raise ValueError("Undefined for sequences of unequal length.")
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    # try to treat as array-like
    xa = np.asarray(a)
    xb = np.asarray(b)
    if xa.shape != xb.shape:
        raise ValueError("Undefined for sequences of unequal shape.")
    return int(np.count_nonzero(xa != xb))


def hamming_distance_normalized(a, b):
    """Compute normalized Hamming distance between two strings or sequences.

    Returns the proportion of positions with differing elements.

    Parameters
    ----------
    a : str or array_like
        First input.
    b : str or array_like
        Second input (must have same length as a).

    Returns
    -------
    float
        Normalized Hamming distance between 0 and 1.

    Raises
    ------
    ValueError
        If sequences have unequal length.

    Examples
    --------
    >>> hamming_distance_normalized("abcd", "abcf")
    0.25
    >>> hamming_distance_normalized("hello", "hello")
    0.0

    Notes
    -----
    The normalized Hamming distance is defined as:

    d(a, b) = Hamming(a, b) / length(a)
    """
    if isinstance(a, str) and isinstance(b, str):
        if len(a) != len(b):
            raise ValueError("Undefined for sequences of unequal length.")
        return sum(c1 != c2 for c1, c2 in zip(a, b)) / len(a)

    xa = np.asarray(a)
    xb = np.asarray(b)
    if xa.shape != xb.shape:
        raise ValueError("Undefined for sequences of unequal shape.")
    return float(np.count_nonzero(xa != xb) / xa.size)


def levenshtein_distance(s1, s2):
    """Compute Levenshtein (edit) distance between two strings.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string
    into another.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    int
        Levenshtein distance.

    Examples
    --------
    >>> levenshtein_distance("kitten", "sitting")
    3
    >>> levenshtein_distance("hello", "hello")
    0
    >>> levenshtein_distance("", "abc")
    3

    Notes
    -----
    Uses dynamic programming with O(n*m) time and O(min(n,m)) space.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein_distance(s1, s2):
    """Compute Damerau-Levenshtein distance between two strings.

    The Damerau-Levenshtein distance is an extension of Levenshtein distance
    that also allows transposition of adjacent characters as a single operation.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    int
        Damerau-Levenshtein distance.

    Examples
    --------
    >>> damerau_levenshtein_distance("ca", "abc")
    2
    >>> damerau_levenshtein_distance("hello", "hello")
    0

    Notes
    -----
    Operations allowed: insertion, deletion, substitution, and transposition
    of adjacent characters.
    """
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)

    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)  # transposition

    return d[(lenstr1 - 1, lenstr2 - 1)]


def jaro_distance(s1, s2):
    """Compute Jaro distance between two strings.

    The Jaro distance is a measure of similarity between two strings.
    The higher the Jaro distance, the more similar the strings.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    float
        Jaro distance between 0 and 1.

    Examples
    --------
    >>> jaro_distance("MARTHA", "MARHTA")
    0.9444444444444445
    >>> jaro_distance("hello", "hello")
    1.0

    Notes
    -----
    The Jaro distance is defined as:

    Jaro(s1, s2) = (1/3) * (m/|s1| + m/|s2| + (m-t)/m)

    where m is the number of matching characters, t is the number of
    transpositions, and |s1|, |s2| are string lengths.
    """
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
    ) / 3

    return float(jaro)


def jaro_winkler_distance(s1, s2, p=0.1):
    """Compute Jaro-Winkler distance between two strings.

    The Jaro-Winkler distance is a modification of Jaro distance that
    gives extra weight to matching prefixes.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.
    p : float, default 0.1
        Scaling factor for prefix matches. Should not exceed 0.25.

    Returns
    -------
    float
        Jaro-Winkler distance between 0 and 1.

    Examples
    --------
    >>> jaro_winkler_distance("MARTHA", "MARHTA")
    0.9666666666666667
    >>> jaro_winkler_distance("hello", "hello")
    1.0

    Notes
    -----
    The Jaro-Winkler distance is defined as:

    JW(s1, s2) = Jaro(s1, s2) + l * p * (1 - Jaro(s1, s2))

    where l is the length of the common prefix (up to 4 characters).
    """
    jaro_sim = jaro_distance(s1, s2)

    # Find length of common prefix (up to 4 characters)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return float(jaro_sim + prefix_len * p * (1 - jaro_sim))