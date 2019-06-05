# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:00:59 2019

@author: kbergste

mms_plotting
exists as module for the mms_feature_search.py main code
"""

from matplotlib.ticker import LogLocator,LogFormatter #for plotting log hists
import matplotlib as mpl
import numpy as np

def tseries_plotter(fig,ax, data1, data2,labels,lims,legend=None):
    '''
    plotter function using matplotlib (mpl) objects
    For timeseries plots ONLY (may generalize in the future)
    Times used are datetime objects
    Inputs:
        fig- the mpl Figure object used
        ax- the mpl Axes object used
        data1- the x-axis variables (numpy array of datetime objects)
        data2- the y-axis variables
        labels- list of three strings, 
            labels[0] is the plot title
            labels[1] is the x label 
            labels[2] is the y label
        lims- list containing the plotting limits for the x axis
            lims[0] is the minimum x-value 
            lims[1] is the maximum x-value
        legend- possible string for the legend of this data. Default is None
    Outputs:
        out- the ax.plot instance used
    '''
    #formatting the x axis and ticks  
    ax.set_xlim(lims[0],lims[1]) 
#    fig.autofmt_xdate() #this helps sometimes but makes things worse sometimes
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M:%S.%f"))
    ax.tick_params(axis='both',length=6,width=2, direction='in' )
    ax.set( title=labels[0], xlabel=labels[1], ylabel=labels[2])
    out = ax.plot(data1, data2,label=legend)
    ax.legend(edgecolor='black')
    return out

def line_maker(axes,time,edges):
    '''
    Makes all horizontal/vertical lines needed on the timeseries plots
    Currently makes a horizontal line at zero and a vertical line at 'time'
    and two more vertical lines at the edge times of the structure
    Times used are datetime objects
    Inputs:
        axes- list-like object of multiple mpl Axes to draw lines on
        time- the central time which will have a red vertical line
        edges- list-like object of the locations of the two blue vertical lines
    
    TODO: Check if the case of a single Axes instance being passed is handled
        correctly
    '''
    for ax in axes:
        ax.axhline(color="black")
        ax.axvline(x=time,color="red")
        ax.axvline(x=edges[0],color="blue")
        ax.axvline(x=edges[1],color="blue")

def histogram_plotter(ax,values,labels,limits,n_bins=10,logscale=False):
    '''
    Makes histograms, for the statistical processing
    Inputs:
        ax- the mpl Axes object used
        values- numpy array for hist
        labels- list of three strings, 
            labels[0] is the plot title
            labels[1] is the x label 
            labels[2] is the y label
        limits- list of [min value, max value]
        n_bins- desired number of bins, default is 10
        logscale- True if want log, False if not, default is False
    Outputs:
        out- the ax.hist instance used
    '''
    
    ax.set( title=labels[0], xlabel=labels[1], ylabel=labels[2])
    ax.set_xlim(limits[0],limits[1])
    
    if logscale:
        ax.set_yscale('log') #try log scale
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatter())
        ax.yaxis.set_minor_locator(LogLocator(subs=(0.2,0.4,0.6,0.8,)))
        ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False,
                                                  minor_thresholds=(2.,5.)))
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(LogFormatter())
        ax.xaxis.set_minor_locator(LogLocator(subs=(0.2,0.4,0.6,0.8,)))
        ax.xaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False,
                                                  minor_thresholds=(2.,5.)))
        log_bins=log_hist_bins(limits,n_bins)
        out=ax.hist(values,log_bins)
    else:
        reg_bins=hist_bins(limits,n_bins)
        out=ax.hist(values,reg_bins)
    
    return out

def log_hist_bins(limits,n_bins):
    '''
    helper function for histogram_plotter to compute evenly spaced bins
    for plotting the log-log histogram.
    inputs:
        limits- list-like object of the minimum and maximum values for the bins
        n_bins- the desired number of bins
    outputs:
        bins- the locations of the bins desired
    '''
    min_val=limits[0]
    max_val=limits[1]
    
    bins=10**np.linspace(np.log10(min_val),np.log10(max_val),n_bins)
    
    return bins

def hist_bins(limits,n_bins):
    '''
    helper function for histogram_plotter to compute evenly spaced bins
    for plotting the regular histogram.
    inputs:
        limits- list-like object of the minimum and maximum values for the bins
        n_bins- the desired number of bins
    outputs:
        bins- the locations of the bins desired
    '''
    min_val=limits[0]
    max_val=limits[1]
    
    bins=np.linspace(min_val,max_val,n_bins)
    
    return bins

def bar_charter(ax,data,labels):
    '''
    Makes bar charts, for summary statistics
    Inputs:
        ax- the mpl Axes object used
        data- a dictionary of dictionaries
            outer dictionary gives the legend groups (e.g. MMS1, MMS2, ...)
            inner dictionary gives the different values for the bars
            (e.g. plasmoid, current sheet)
            (MUST be the same keys over all groups!)
        labels- list of three strings, 
            labels[0] is the plot title
            labels[1] is the x label 
            labels[2] is the y label
    Outputs:
        data_bars- list of the different bars of the bar chart        
    '''
    spacing=0.2  #desired distance between bars of different groups
    legends=list(data.keys())
    ind=np.arange(len(data[legends[0]]))  #number of indices
    width=(1. - spacing)/len(data[legends[0]])  #width of each bar
    data_bars=[]  #list of the parts of the bar chart
    
    for i,dataset in enumerate(legends):
        rect=ax.bar(ind-(0.5-spacing/2.)+i*width, data[dataset].values(), 
                    width, label=dataset)
        data_bars.append(rect)
        
    ax.legend(edgecolor='black')
    ax.set(title=labels[0],xlabel=labels[1],ylabel=labels[2])
    ax.set_xticks(ind)
    ax.set_xticklabels(tuple(data[legends[0]].keys()))
    
    return data_bars

