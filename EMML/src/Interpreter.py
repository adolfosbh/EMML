'''
Created on 17 Apr 2014

@author: asbh500
'''

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    y1=[125,115,130,140,140,115,140,125,140,135]
    y2=[110,122,125,120,140,124,123,137,135,145]

    print ss.mannwhitneyu(y1, y2)
    print ss.mannwhitneyu(y2, y1)
    print ss.mannwhitneyu(y2, list(reversed(y1)))
    print ss.mannwhitneyu(y1, list(reversed(y2)))



    plt.show()

    #print ss.wilcoxon(y1, y2, zero_method="zsplit")
    #print ss.wilcoxon(y1, y2, zero_method="pratt")
    #print ss.wilcoxon(y1, y2, zero_method="wilcox")
    #print ss.wilcoxon(y1, y2, zero_method="zsplit", correction=True )
    #print ss.wilcoxon(y1, y2, zero_method="pratt",  correction=True )
    #print ss.wilcoxon(y1, y2, zero_method="wilcox",  correction=True )


