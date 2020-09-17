#!/usr/bin/env python
# coding: utf-8

from itertools import islice

def contiguous_voxels(seq, window_size):
    #remove non-contiguous voxels
    
    window_total = 0
    list1 = []
    list2 = []
    list3 = []
    revised_list = []
    
    #perform sliding window fcn
    windows = list(sliding_window(seq, window_size))
    
    for lis in windows:
        window_total = lis[0] + lis[1] + lis[2]
        if lis[0] + (lis[0] + 1) + (lis[0] + 2) == window_total:
            list1.append(lis)

    list2 = [lis[0] for lis in list1]
    list3 = [lis[1] for lis in list1]
    list4 = [lis[2] for lis in list1]

    list5 = list2 + list3 + list4
    revised_list = list(dict.fromkeys(list5))
    revised_list.sort()
    
    return(revised_list)
	
def sliding_window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    
    it = iter(seq)
    result = tuple(islice(it, n))
    
    if len(result) == n:
        yield result
    
    for elem in it:
        result = result[1:] + (elem,)
        yield result