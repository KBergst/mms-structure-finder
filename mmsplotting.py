# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:00:59 2019

@author: kbergste

mms_plotting
exists as module for the mms_feature_search.py main code
"""

from matplotlib.ticker import LogLocator,LogFormatter #for plotting log hists
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import re #for url-ifying strings
import textwrap #for fixing too-long labels that overlap
from scipy.optimize import curve_fit #for fitting histograms
import sklearn as sk
import scipy.stats as stats
import statsmodels.api as sm

##### FITTING FNS #############################################################
def fitfn_linear(x,a,b):
    '''
    fitting function
    linear function
    fits for parameters a and b
    '''
    linear_fn=a*x+b
    
    return linear_fn

def fitfn_exp(x,a,b):
    '''
    fitting function (specifically for size histograms)
    decaying exponential
    fits for parameters a and b
    Inputs/outputs fairly obvious
    '''
    exp_fn=a*np.exp(-b*x)
    
    return exp_fn

def fitfn_pwr(x,a,b):
    '''
    fitting function (specifically for size histograms)
    power law
    fits for parameters a and b  
    Inputs/outputs fairly obvious
    '''
    pwr_fn=a*x**b
    
    return pwr_fn

def size_fitter(x,y,fitfn,guess,yerrs=None):
    '''
    wrapper for fitting the size histograms 
    Inputs:
        x- the independent variable data to be fit
        y- the dependent variable data to be fit (same dimension as x)
        fitfn- the fitting function to be used
        guess- guess for the fit parameters (depend on the fitting function used)
        yerrs- error on the y variables. Default None
    Outputs:
        x_smooth- the fit independent variables
        y_smooth- the fit dependent variables
        popt- the fit parameters
        errs- the standard deviations of the fit parameters
        errbars- error bars for each x_for_fit point (bundled with the x and y points at which they exist)
    '''
    #section the data for the fit- want the part above the max fitted for size
    max_idx=np.argmax(y)
    x_for_fit=x[max_idx:]
    y_for_fit=y[max_idx:]
    yerrs_for_fit=yerrs[max_idx:]
    popt, pcov = curve_fit(fitfn,xdata=x_for_fit,ydata=y_for_fit,
                           p0=guess,sigma=yerrs_for_fit)
    fit_bdy=[x_for_fit[0],x_for_fit[-1]]
    x_smooth=np.linspace(fit_bdy[0],fit_bdy[1],num=1000)
    y_smooth=fitfn(x_smooth,*popt)
    errs=np.sqrt(np.diagonal(pcov)) #returns standard deviations
    
    errbars=errbar_maker(x_for_fit,fitfn,popt,errs)
    y_for_errs=fitfn(x_for_fit,*popt)
#    y_for_c2 = y_for_fit[0:min(5,y_for_fit.size)]
#    yexp_for_c2 = y_for_errs[0:y_for_c2.size]
#    c2,p=chisquare(y_for_c2,yexp_for_c2)
    
    return x_smooth,y_smooth,popt,errs,[x_for_fit,y_for_errs,errbars]

def mle_fitter(y,distfn,**kwargs):
    '''
    Finds Maximum Likelihood Estimate for the sample y given distribution function distfn.
    Leaves space for doing fancy stuff later if desired
    '''
    
    distparams=distfn.fit(y,**kwargs)
    ysmooth=np.linspace(min(y),max(y),900)
    pdfsmooth=distfn.pdf(ysmooth,*distparams)
    
    return distparams,ysmooth,pdfsmooth
    

def errbar_maker(x,fitfn,params,param_errs):
    '''
    Creates error bars for the histogram fits
    Inputs:
        x- the x locations of the center of each histogram bar
        fitfn- the fitting function used
        params- the parameters for fitting function returned by curve_fit
        param_errs- the standard deviations on the parameters returned by curve_fit
    '''
    min_params=params-param_errs
    max_params=params+param_errs
    
    fit_val=fitfn(x,params[0],params[1])
    val_min=fitfn(x,min_params[0],min_params[1])    
    val_max=fitfn(x,max_params[0],max_params[1])
    errbar_min=fit_val-val_min
    errbar_max=val_max-fit_val
    
    errbars=np.row_stack((errbar_min,errbar_max))
    
    return errbars

### truncated power law distribution ################
class powerlw(stats.rv_continuous):
    "truncated power law distribution - has lower and upper cutoffs (to match data better)"
    
    def _pdf(self,x,p):
        a=self.a
        b=self.b
        return (1-p)/(b**(1-p)-a**(1-p))*x**(-1*p)
    
    def _cdf(self,x,p):
        a=self.a
        b=self.b
        return (x**(1-p)-a**(1-p))/(b**(1-p)-a**(1-p))
        
    
    
###### DIRECTORY MANAGEMENT FNS ###############################################
def directory_ensurer(directory):
    '''
    Ensures that the directory specified by the given path exists
    Important for initializing the code on a new machine, since folders
    that only contain output files which are not tracked by git (.png)
    will not be initially present in the github file
    Inputs:
        directory- string containing full path to the desired directory
    Outputs:
        none
    '''
    os.makedirs(directory, exist_ok=True)
    
def urlify(string):
    '''
    Taken from a stack overflow post on how to replace whitespaces in strings
    and generally make strings more url-compatible
    https://stackoverflow.com/questions/1007481/how-do-i-replace-whitespaces-
    with-underscore-and-vice-versa
    Documentation of the 're' module:
        https://docs.python.org/3/library/re.html
    Inputs:
        string- the string to be made compatible for URL purposes
    Outputs:
        url_string- string without special characters or white spaces
    '''
    # Remove all non-word characters (all non-alphanumeric)
    s_numlett=re.sub(r"[^\w\s]", '', string)
    
    # Replace all whitespace with an underscore
    s_url = re.sub(r"\s+", '_', s_numlett)
    
    return s_url

def textify(string):
    '''
    want to replace underscores with spaces for display purposes
    Inputs:
        string- the string (likely an attribute or other variable name) to
            change to plain text
    Outputs:
        s_plaintext- the string in plain text for plot titles, etc.
    '''
    
    #Replace all underscores and dashes with spaces
    s_plaintext=re.sub(r"[_-]",' ', string)
    return s_plaintext

##### PLOTTING FNS ############################################################
def basic_plotter(ax,data1,data2,equalax=False,legend=None,labels=None,
                  yerrors=None,square=False,ylims=None,colorval=None):
    '''
    plotter function using matplotlib (mpl) objects
    For timeseries plots ONLY (may generalize in the future)
    Times used are datetime objects
    Inputs:
        fig- the mpl Figure object used
        ax- the mpl Axes object used
        data1- the x-axis variables
        data2- the y-axis variables
        legend- possible string for the legend of this data. Default is None
        labels- list of three strings, 
            labels[0] is the plot title
            labels[1] is the x label 
            labels[2] is the y label
            default is None
        y errors- the y error bars for each point. Default None, can either be
            a constant value or have the same dimension as data1 and data2
        square- true if want square plot, false if not. Default False
        ylims- list containing the plotting limits for the y axis
            lims[0] is the minimum y-value 
            lims[1] is the maximum y-value
        colorval- for specifying color, default None
    Outputs:
        out- the ax.plot instance used
    '''
    out = ax.plot(data1, data2,label=legend,color=colorval)
    if not (yerrors is None): #set legend, if it exists
        ax.errorbar(data1,data2,yerr=yerrors,fmt='.',capsize=3)
    if not (labels is None): #set labels, if they exist
        ax.set( title=labels[0], xlabel=labels[1], ylabel=labels[2])
    if equalax is True:
        ax.set_aspect('equal')
    if square is True:
        xlims,ylims=window_squarer(data1,data2)
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
    ax.legend(edgecolor='black')
    
    if (ylims is not None):
        ax.set_ylim(ylims[0],ylims[1])
        
    return out

def window_squarer(data1,data2):
    '''
    makes the window have the same x and y limits (thus makes it a square,
                                                   if the axes are equal)
    Inputs:
        data1- the x-axis variables
        data2- the y-axis variables
    Outputs:
        xlims- list of the minimum and maximum values for the x axis
        ylims- list of the minimum and maximum values for the y axis
    '''
    scale1=np.amax(data1)-np.amin(data1)
    mid1=np.amin(data1)+scale1/2
    scale2=np.amax(data2)-np.amin(data2)
    mid2=np.amin(data2)+scale2/2
    scale_tot=max([scale1,scale2])
    
    xlims=[mid1-scale_tot/2,mid1+scale_tot/2]
    ylims=[mid2-scale_tot/2,mid2+scale_tot/2]
    
    return xlims,ylims
    
def tseries_plotter(fig,ax, data1, data2,labels,lims,legend=None,
                    logscale=False):
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
        logscale- display y axis on a log scale. Default is False.
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
    if not (legend is None):
        ax.legend(edgecolor='black')
    
    if logscale:
        ax.set_yscale('log') #try log scale
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatter())
        ax.yaxis.set_minor_locator(LogLocator(subs=(0.2,0.4,0.6,0.8,)))
        ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False,
                                                  minor_thresholds=(2.,5.)))
        
    return out

def line_maker(axes,horiz=None,time=None,edges=None):
    '''
    Makes all horizontal/vertical lines needed on the timeseries plots
    Currently makes a horizontal line at zero and a vertical line at 'time'
    and two more vertical lines at the edge times of the structure
    Times used are datetime objects
    Inputs:
        axes- list-like object of multiple mpl Axes to draw lines on
        horiz- the location of the horizontal line to be drawn, default None
        time- the central time which will have a red vertical line. Default None
        edges- list-like object of the locations of the two blue vertical lines
            default None
    
    TODO: Check if the case of a single Axes instance being passed is handled
        correctly
    '''
    for ax in axes:
        if not (horiz is None):
            ax.axhline(y=horiz,color="black")            
        if not (time is None):
            ax.axvline(x=time,color="red")
        if not (edges is None):
            ax.axvline(x=edges[0],color="blue")
            ax.axvline(x=edges[1],color="blue")

def histogram_plotter(ax,values,labels,limits,n_bins=10,logscale=False):
    '''
    Makes density histograms, for the statistical processing
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
        errs- the error bars on the plots
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
        errs = hist_errs(ax,out)
    else:
        reg_bins=hist_bins(limits,n_bins)
        out=ax.hist(values,reg_bins)
        errs = hist_errs(ax,out)
    
    return [out,errs]

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

def hist_errs(ax,out):
    '''
    Creates standard poisson error bars for a given histogram,
    Using bin_err=sqrt(bin_size)
    Inputs:
        ax- the axes on which to plot the error bars
        out- the output from the generated histogram. 
            out[0] the array of values of the histogram bins
            out[1] the edges of the bins
    '''
    errs=[]
    for i,bar_amt in enumerate(out[0]):
        min_err = 1
        top_err = max(min_err,np.sqrt(bar_amt)) #set errors to 1 if nothing in the bar
        bottom_err=np.sqrt(bar_amt) #bottom error is zero if nothing in bar       
        errs.append([top_err])
        errs_point=np.array([bottom_err,top_err])
        errs_point = errs_point.reshape((2,1))
        bar_center = (out[1][i+1]+out[1][i])/2
        ebar_plotter(ax,bar_center,bar_amt,errs_point,colorval='black')
        
    errs_arr=np.concatenate(errs)
    return errs_arr

def ebar_plotter(ax,data1,data2,errbars,colorval):
    '''
    function to plot error bars only on specific points
    useful if there are only errors on some of the points
    Inputs:
        ax- the axis to use
        data1- the x-axis variables
        data2- the y-axis variables
        errbars- array of errors, in proper matplotlib format (see documentation)
        colorval- matplotlib color arg for the lines
    Outputs:
        out- the ax.errorbar instance used
    '''
    
    out=ax.errorbar(data1,data2,yerr=errbars,fmt='.',capsize=3,ecolor=colorval,
                    color=colorval)
    
    return out

def scatter_plotter(ax,data1,data2,labels,marker_type=None,legend=None):
    '''
    Makes scatter plots using matplotlib objects
    Inputs:
        ax- matplotlib Axes object to be plotted on
        data1- the x-axis variables
        data2- the y-axis variables
        labels- list of three strings, 
            labels[0] is the plot title
            labels[1] is the x label 
            labels[2] is the y label
        marker_type- string naming the marker style
        legend- possible string for the legend of this data. Default is None
    Outputs:
        out- the ax.scatter instance used
    '''
    

    ax.set( title=labels[0], xlabel=labels[1], ylabel=labels[2])
    out = ax.scatter(data1, data2,marker=marker_type,label=legend)
    ax.legend(edgecolor='black')
    
    return out
    
def bar_charter(ax,data,labels):
    '''
    Makes bar charts, for summary statistics
    Inputs:
        ax- the mpl Axes object used
        data- a dictionary of dictionaries
            outer dictionary gives the legend groups (e.g. MMS1, MMS2, ...)
                if no desired legend, outer dictionary contains a blank string
                as a key
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
    width=(1. - spacing)/len(legends)  #width of each bar
    data_bars=[]  #list of the parts of the bar chart
    
    for i,dataset in enumerate(legends):
        if dataset is '':
            legend=None
        else:
            legend=dataset
            
        rect=ax.bar(ind-(0.5-spacing/2.)+i*width, data[dataset].values(), 
                    width, label=legend)
        data_bars.append(rect)
        
    ax.legend(edgecolor='black')
    ax.set(title=labels[0],xlabel=labels[1],ylabel=labels[2])
    ax.set_xticks(ind)

    xlabels=['\n'.join(textwrap.wrap(l,10)) for l in data[legends[0]].keys()]
    ax.set_xticklabels(xlabels)
    
    return data_bars

def pie_plotter(ax,amounts,labels,title):
    '''
    Makes a pie chart of the inputs
    Inputs:
        ax- the matplotlib axes on which to plot the pie
        amounts- a list or array of the various amounts for the pie chart
        labels- the labels for each part of the pie, in the same order as amounts
        title- the title of the plot
        **kwargs- dictionary of keyword arguments for the pie chart
    Outputs:
        out- the output of the plot
    '''
    
    wedges, texts= ax.pie(amounts)
    ax.legend(wedges,labels)
    ax.set_title(title)
    
    return wedges
    
def structure_hist_maker(data,attr,out,bins_num,structure_key,
                         log=False):
    '''
    A specialized function for plotting histograms of various structure
    attributes.
    Inputs:
        data-A dictionary with keys of the satellites and 
            values of arrays of structures.
        attr- a string naming the desired attribute being plotted 
            (e.g. size)- MUST be a float valued attribute!
        out- A string containing the desired output location
        bins_num- number of bins desired for the histogram
        structure_key- list of the different structure types, as strings
        log-True if want log scale, False if not, default is False
    Outputs:
        no output (void)
        Writes the histograms to a file at the given output location
    '''
    #extract information about the units and plurals of the desired attribute
    struct_ex=data[list(data.keys())[0]][0] #returns first structure in array
    attrs=struct_ex.plurals[attr]
    attr_units=struct_ex.units[attr]
    attr_txt=textify(attr)
    #make histograms
    for structure_type in structure_key:
        #structure data
        outs=[]
        labels_tot=['{} of {} over all satellites'.format(attrs.capitalize(),
                    structure_type),
                    '{} ({})'.format(attr_txt.capitalize(),
                                     attr_units),
                    'Number of instances']
        labels_M=[]
        for i,M in enumerate(list(data.keys())):
            labels_M.append(['{} of {} for MMS {}'.format(attrs.capitalize(),
                             structure_type,M),
                               '{} ({})'.format(attr_txt.capitalize(),
                                                attr_units),
                               'Number of instances'])
        
        total_data=np.array([])
        sat_data={} #for extracting data per satellite
        for sat in list(data.keys()):
            tmp=np.vectorize(lambda x: getattr(x,attr)) \
                                                        (data[sat])
            struct_mask=np.vectorize(lambda x: x.kind == structure_type) \
                                                        (data[sat])
            sat_data[sat]= tmp[struct_mask]                                           
            total_data=np.append(total_data,sat_data[sat])
        
        all_limits=[min(total_data),max(total_data)]
        #plot everything    
        mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
        plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
        gridsize=(5,1)
        fig=plt.figure(figsize=(8,12)) #width,height
        ax1=plt.subplot2grid(gridsize,(0,0))
        ax2=plt.subplot2grid(gridsize,(1,0))   
        ax3=plt.subplot2grid(gridsize,(2,0)) 
        ax4=plt.subplot2grid(gridsize,(3,0))
        ax5=plt.subplot2grid(gridsize,(4,0))
        
        outs.append(histogram_plotter(ax1,sat_data['1'],labels_M[0],all_limits,
                          n_bins=bins_num,logscale=log))
        outs.append(histogram_plotter(ax2,sat_data['2'],labels_M[1],all_limits,
                          n_bins=bins_num,logscale=log))
        outs.append(histogram_plotter(ax3,sat_data['3'],labels_M[2],all_limits,
                          n_bins=bins_num,logscale=log))
        outs.append(histogram_plotter(ax4,sat_data['4'],labels_M[3],all_limits,
                          n_bins=bins_num,logscale=log))
        outs.append(histogram_plotter(ax5,total_data,labels_tot,all_limits,
                                      n_bins=bins_num,logscale=log))
        axs=[ax1,ax2,ax3,ax4,ax5] #for doing the fits
        if (attr == 'size' and (structure_type == 'plasmoids' \
                                or structure_type == 'pull current sheets' \
                                or structure_type == 'push current sheets')): #do fitting
            arrays=[item[0][1] for item in outs]
            bin_edges=[item[0][1] for item in outs]
            yerrs=[item[1] for item in outs]
            for i,(arr,bins,yerr) in enumerate(zip(arrays,bin_edges,yerrs)):
                bin_centers=np.array([0.5 * (bins[i] + bins[i+1]) \
                                      for i in range(len(bins)-1)])
                pwr_guess=[max(arr),-1.]
                exp_guess=[max(arr),0.005]
                x_exp,y_exp,params_exp,exp_errs,exp_ebars,exp_c2=size_fitter(bin_centers,
                                                                      arr,
                                                            fitfn_exp,
                                                            exp_guess,
                                                            yerrs = yerr)
                x_pwr,y_pwr,params_pwr,pwr_errs,pwr_ebars,pwr_c2=size_fitter(bin_centers,arr,
                                                            fitfn_pwr,
                                                            pwr_guess,
                                                            yerrs = yerr)
                basic_plotter(axs[i],x_exp,y_exp,
                              legend=r'Exponential fit $({}\pm {})e^{{-({}\pm {})x}}$' \
                              .format(f"{params_exp[0]:.2f}",
                                         f"{exp_errs[0]:.2f}",
                                         f"{params_exp[1]:.4f}",
                                         f"{exp_errs[1]:.4f}"))
                basic_plotter(axs[i],x_pwr,y_pwr,
                              legend='Power law fit $({}\pm {}) x^{{ ({}\pm {}) }}$' \
                              .format(f"{params_pwr[0]:.2f}",
                                         f"{pwr_errs[0]:.2f}",
                                         f"{params_pwr[1]:.2f}",
                                         f"{pwr_errs[1]:.2f}")) 
        if log:
            structure_type+='_log'
            
        fig.savefig(os.path.join(out,"{}_hist_{}.png".format(attr,
                                 urlify(structure_type))), bbox_inches='tight')
        plt.close(fig='all')

    ''' do overall plot '''
    outs=[]
    labels_tot=['{} of all structure types over all satellites' \
                .format(attrs.capitalize()), 
                '{} ({})'.format(attr_txt.capitalize(),attr_units),
                'Number of instances']
    labels_M=[]
    for i,M in enumerate(list(data.keys())):
        labels_M.append(['{} of all structure types for MMS {}'\
                         .format(attrs.capitalize(),M),
                '{} ({})'.format(attr_txt.capitalize(),attr_units),
                'Number of instances'])
    sat_data={} #for extracting data per satellite
    for sat in list(data.keys()):
        sat_data[sat]=np.vectorize(lambda x: getattr(x,attr)) \
                                                    (data[sat])                                       
        total_data=np.append(total_data,sat_data[sat])
    all_limits=[min(total_data),max(total_data)]
    #plot everything    
    mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
    plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
    gridsize=(5,1)
    fig=plt.figure(figsize=(8,12)) #width,height
    ax1=plt.subplot2grid(gridsize,(0,0))
    ax2=plt.subplot2grid(gridsize,(1,0))   
    ax3=plt.subplot2grid(gridsize,(2,0)) 
    ax4=plt.subplot2grid(gridsize,(3,0))
    ax5=plt.subplot2grid(gridsize,(4,0))
    
    outs.append(histogram_plotter(ax1,sat_data['1'],labels_M[0],all_limits,
                      n_bins=bins_num,logscale=log))
    outs.append(histogram_plotter(ax2,sat_data['2'],labels_M[1],all_limits,
                      n_bins=bins_num,logscale=log))
    outs.append(histogram_plotter(ax3,sat_data['3'],labels_M[2],all_limits,
                      n_bins=bins_num,logscale=log))
    outs.append(histogram_plotter(ax4,sat_data['4'],labels_M[3],all_limits,
                      n_bins=bins_num,logscale=log))
    outs.append(histogram_plotter(ax5,total_data,labels_tot,all_limits,n_bins=bins_num,
                      logscale=log))
    axs=[ax1,ax2,ax3,ax4,ax5] #for doing the fits
    
#    if (attr == 'size'): #do fitting
#        arrays=[item[0] for item in outs]
#        bin_edges=[item[1] for item in outs]
#        for i,(arr,bins) in enumerate(zip(arrays,bin_edges)):
#            bin_centers=np.array([0.5 * (bins[i] + bins[i+1]) \
#                                  for i in range(len(bins)-1)])
#            x_exp,y_exp,params_exp=size_fitter(bin_centers,arr,fitfn_exp)
#            x_pwr,y_pwr,params_pwr=size_fitter(bin_centers,arr,fitfn_pwr)
#            basic_plotter(axs[i],x_exp,y_exp,
#                          legend='Exponential fit ${} e^{{-{}x}}$' \
#                          .format(f"{params_exp[0]:.2f}",
#                                         f"{params_exp[1]:.2f}"))
#            basic_plotter(axs[i],x_pwr,y_pwr,
#                          legend='Power law fit ${} x^{{ {} }}$' \
#                          .format(f"{params_pwr[0]:.2f}",
#                                         f"{params_pwr[1]:.2f}")) 
    
    suffix=''
    if log:
        suffix='_log'
        
    fig.savefig(os.path.join(out,"{}_hist_all_structures{}.png".format(attr,
                             suffix)), 
                bbox_inches='tight')
    plt.close(fig='all')
    
def structure_scatter_maker(data,attr1,attr2,out,structure_key,type_strs=None):
    '''
    A specialized function for plotting scatter plots of various structure
    attributes against each other
    Inputs:
        data-A dictionary with keys of the satellites and 
            values of arrays of structures.
        attr1- a string naming the desired x-axis attribute being plotted 
            (e.g. size)- MUST be a float valued attribute!
        attr2- a string naming the desired y-axis attribute being plotted 
            (e.g. size)- MUST be a float valued attribute!
        out- A string containing the desired output location
        structure_key- list of the different structure types, as strings
    Outputs:
        no output (void)
        Writes the histograms to a file at the given output location
    '''
    #extract information about the units and plurals of the desired attributes
    struct_ex=data[list(data.keys())[0]][0] #returns first structure in array
    attr1s=struct_ex.plurals[attr1]
    attr2s=struct_ex.plurals[attr2]
    attr1_units=struct_ex.units[attr1]
    attr2_units=struct_ex.units[attr2]
    attr1_txt=textify(attr1)
    attr2_txt=textify(attr2)
    #make scatter plots
    for structure_type in structure_key:
        #structure data
        labels_tot=['{} versus {} of {}' \
                    .format(attr2s.capitalize(),attr1s.capitalize(),
                            structure_type),
                    '{} ({})'.format(attr1_txt.capitalize(),
                                     attr1_units),
                    '{} ({})'.format(attr2_txt.capitalize(),
                                     attr2_units)]
        
        #get plot ready
        mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
        plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
        fig,ax=plt.subplots(figsize=(6,6))
        markers=['o','v','^','s'] #choose markers for each satellite
        
        sat_data1={} #for extracting data per satellite
        sat_data2={}
        for i,sat in enumerate(list(data.keys())):
            tmp1=np.vectorize(lambda x: getattr(x,attr1)) \
                                                        (data[sat])
            tmp2=np.vectorize(lambda x: getattr(x,attr2)) \
                                                        (data[sat])
            struct_mask=np.vectorize(lambda x: x.kind == structure_type) \
                                                        (data[sat])
            sat_data1[sat]= tmp1[struct_mask]
            sat_data2[sat]= tmp2[struct_mask]     
            
            scatter_plotter(ax,sat_data1[sat],sat_data2[sat],labels_tot,
                            marker_type=markers[i],legend=sat)
            
        fig.savefig(os.path.join(out,"{}_{}_scatter_{}.png".format(attr1,attr2,
                                 urlify(structure_type))), bbox_inches='tight')
        plt.close(fig='all')

    ''' do overall plot '''
    labels_tot=['{} versus {} of all structure types' \
                .format(attr2s.capitalize(),attr1s.capitalize()),
                '{} ({})'.format(attr1_txt.capitalize(),
                                 attr1_units),
                '{} ({})'.format(attr2_txt.capitalize(),
                                 attr2_units)]

    #get plot ready
    mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
    plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
    fig,ax=plt.subplots(figsize=(6,6))
    markers=['o','v','^','s'] #choose markers for each satellite
    
    sat_data1={} #for extracting data per satellite
    sat_data2={}
    for i,sat in enumerate(list(data.keys())):
        sat_data1[sat]=np.vectorize(lambda x: getattr(x,attr1)) \
                                                    (data[sat])
        sat_data2[sat]=np.vectorize(lambda x: getattr(x,attr2)) \
                                                    (data[sat])
                                                    
        scatter_plotter(ax,sat_data1[sat],sat_data2[sat],labels_tot,
                marker_type=markers[i],legend=sat)
        
    fig.savefig(os.path.join(out,"{}_{}_scatter_all_structures.png" \
                             .format(attr1,attr2)),
                bbox_inches='tight')
    plt.close(fig='all')

def msc_structure_hist_maker(data,attr,out,bins_num,structure_key,
                         log=False):
    '''
    A specialized function for plotting histograms of various structure
    attributes for the multispacecraft script.
    Inputs:
        data-A list of structures.
        attr- a string naming the desired attribute being plotted 
            (e.g. size)- MUST be a float valued attribute!
        out- A string containing the desired output location
        bins_num- number of bins desired for the histogram
        structure_key- list of the different structure types, as strings
        log-True if want log scale, False if not, default is False
    Outputs:
        no output (void)
        Writes the histograms to a file at the given output location
    '''
    #extract information about the units and plurals of the desired attribute
    struct_ex=data[0] #returns first structure in array
    attrs=struct_ex.plurals[attr]
    attr_units=struct_ex.units[attr]
    attr_txt=textify(attr)
    #make histograms
    for structure_type in structure_key:
        #structure data
        labels=['{} of {} over all satellites'.format(attrs.capitalize(),
                    structure_type),
                    '{} ({})'.format(attr_txt.capitalize(),
                                     attr_units),
                    'Number of instances']        
        total_data=np.array([])
        for structure in data:
            if structure.kind == structure_type:
                structure_dat=getattr(structure,attr)                                          
                total_data=np.append(total_data,structure_dat)
        
        all_limits=[min(total_data),max(total_data)]
        #plot everything    
        mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
        plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
        fig,ax=plt.subplots()
        
        (vals,bins,tmp),histerrs=histogram_plotter(ax,total_data,labels,all_limits,
                          n_bins=bins_num,logscale=log)
        if ((attr == 'size' or attr == 'normal_speed') \
                                    and (structure_type == 'plasmoids' \
                                or structure_type == 'pull current sheets' \
                                or structure_type == 'push current sheets')): #do fitting
            area=sum(np.diff(bins)*vals)
#            pwr_guess=[max(arr),-1.]
#            exp_guess=[max(arr),0.005]
            #do fitting
            pwr=powerlw(a=min(total_data),b=max(total_data))
            params_exp,x_exp,y_exp=mle_fitter(total_data,stats.expon)
            params_pwr,x_pwr,y_pwr=mle_fitter(total_data,pwr)
            #do KS testing
            ks_exp=stats.kstest(total_data,'expon',args=params_exp) #on all data points

            ks_pwr=stats.kstest(total_data,pwr.cdf,args=params_pwr) #on all data points
            
            #do bootstrapping shittily
            n_samples=100
            exp_coeff1=[]
            exp_coeff2=[]
            exp_coeff3=[]
            pwr_coeff1=[]
            pwr_coeff2=[]
            pwr_coeff3=[]
            for i in range(n_samples):
                y_sample=sk.utils.resample(total_data,replace=True)
                
                sample_expfit=stats.expon.fit(y_sample)
                exp_coeff1.append(area/sample_expfit[1])
                exp_coeff2.append(sample_expfit[0])
                exp_coeff3.append(sample_expfit[1])
                
                sample_pwr=powerlw(a=min(y_sample),b=max(y_sample))
                sample_pwrfit=sample_pwr.fit(y_sample)
                pwr_coeff1.append(area*sample_pwrfit[2]**(sample_pwrfit[0]-1))
                pwr_coeff2.append(sample_pwrfit[1])
                pwr_coeff3.append(sample_pwrfit[0])
                
## plotting that includes errors
#            basic_plotter(ax,x_exp,y_exp,
#                          legend=r'Exponential fit $({}\pm {})e^{{-({}\pm {})x}}$'
#                          r' KS p-value = {}'.format(f"{params_exp[0]:.2f}",
#                                     f"{exp_errs[0]:.2f}",
#                                     f"{params_exp[1]:.4f}",
#                                     f"{exp_errs[1]:.4f}",
#                                     f"{ks_exp[1]:.2f}"), colorval="blue")
#            basic_plotter(ax,x_pwr,y_pwr,
#                          legend=r'Power law fit $({}\pm {}) x^{{ ({}\pm {}) }}$'
#                          r' KS p-value = {}'.format(f"{params_pwr[0]:.2f}",
#                                     f"{pwr_errs[0]:.2f}",
#                                     f"{params_pwr[1]:.2f}",
#                                     f"{pwr_errs[1]:.2f}",
#                                     f"{ks_pwr[1]:.2f}"), colorval="red") 
            ## plotting not including errors
            basic_plotter(ax,x_exp,area*y_exp,
                          legend=r'Exponential distribution $Ae^{{(b+x)/c }}$'+'\n'
                                     r'A={}$\pm${}, b={}$\pm${}, c={}$\pm${}'
                                   r' KS p-value = {}'.format(f"{area/params_exp[1]:.1f}",f"{np.std(exp_coeff1):.1f}",
                                     f"{params_exp[0]:.1f}",f"{np.std(exp_coeff2):.1f}",
                                     f"{params_exp[1]:.1f}",f"{np.std(exp_coeff3):.1f}",
                                     f"{ks_exp[1]:.2f}"),colorval="blue")
            basic_plotter(ax,x_pwr,area*y_pwr,
                          legend=r'Power law distribution $A(b+x)^{{-c }}$'+'\n'
                                  r'A={} $\pm$ {}, b={}$\pm${}, c={}$\pm${}'
                                   r' KS p-value = {}'.format(f"{area*params_pwr[2]**(params_pwr[0]-1):.1e}",f"{np.std(pwr_coeff1):.1e}",
                                     f"{params_pwr[1]:.1f}",f"{np.std(pwr_coeff2):.1f}",
                                     f"{params_pwr[0]:.2f}",f"{np.std(pwr_coeff3):.2f}",
                                     f"{ks_pwr[1]:.2f}"), colorval="red")
            
            #make ppplots and save separately
            figpp,axpp=plt.subplots(2)
            #Probplot objects
            pp_exp=sm.ProbPlot(total_data,'expon',loc=params_exp[0],scale=params_exp[1])
            pp_pwr=sm.ProbPlot(total_data,pwr,loc=params_pwr[1],scale=params_pwr[2],
                               distargs=(params_pwr[0],))
            pp_exp.ppplot(line='45',ax=axpp[0])
            pp_pwr.ppplot(line='45',ax=axpp[1])
            axpp[0].set(title="Probability-probability plot against exponential distribution")
            axpp[1].set(title="Probability-probability against power-law distribution")
#            axqq[0].set_xscale("log")
#            axqq[0].set_yscale("log")
#            axqq[1].set_xscale("log")
#            axqq[1].set_yscale("log")
            
            qqsuffix=''
            if log:
                qqsuffix='log'
            figpp.savefig(os.path.join(out,"{}_pplot{}{}.png".format(attr,urlify(structure_type),
                                       qqsuffix)),
                          bbox_inches='tight')
            
            

        if log:
            structure_type+='_log'
            
        fig.savefig(os.path.join(out,"{}_hist_{}.png".format(attr,
                                 urlify(structure_type))), bbox_inches='tight')
        plt.close(fig='all')

def msc_structure_scatter_maker(data,attr1,attr2,out,structure_key):
    '''
    A specialized function for plotting scatter plots of various structure
    attributes against each other
    Inputs:
        data- A list of structures.
        attr1- a string naming the desired x-axis attribute being plotted 
            (e.g. size)- MUST be a float valued attribute!
        attr2- a string naming the desired y-axis attribute being plotted 
            (e.g. size)- MUST be a float valued attribute!
        out- A string containing the desired output location
        structure_key- list of the different structure types, as strings
    Outputs:
        no output (void)
        Writes the histograms to a file at the given output location
    '''
    #extract information about the units and plurals of the desired attributes
    struct_ex=data[0] #returns first structure in array
    attr1s=struct_ex.plurals[attr1]
    attr2s=struct_ex.plurals[attr2]
    attr1_units=struct_ex.units[attr1]
    attr2_units=struct_ex.units[attr2]
    attr1_txt=textify(attr1)
    attr2_txt=textify(attr2)
    #make scatter plots
    for structure_type in structure_key:
        #structure data
        labels_tot=['{} versus {} of {}' \
                    .format(attr2s.capitalize(),attr1s.capitalize(),
                            structure_type),
                    '{} ({})'.format(attr1_txt.capitalize(),
                                     attr1_units),
                    '{} ({})'.format(attr2_txt.capitalize(),
                                     attr2_units)]
        
        #get plot ready
        mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
        plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
        fig,ax=plt.subplots(figsize=(6,6))
        total_data1=np.array([])
        total_data2=np.array([])
        for structure in data:
            if structure.kind == structure_type:
                data1=getattr(structure,attr1)   
                data2=getattr(structure,attr2)
                total_data1=np.append(total_data1,data1)
                total_data2=np.append(total_data2,data2)
                
        scatter_plotter(ax,total_data1,total_data2,labels_tot)
        #plot the artificial limits on sizes/velocities for size and velocity scatter
        if (('size' in [attr1,attr2]) and ('normal_speed' in [attr1,attr2])):
            x=np.linspace(0,max(total_data1),num=10)
            y1=fitfn_linear(x,128,0)
            y2=fitfn_linear(x,1/3,0)
            y_lims=[0,max(total_data2)*1.05]
            basic_plotter(ax,x,y1,ylims=y_lims)
            basic_plotter(ax,x,y2)

            
        fig.savefig(os.path.join(out,"{}_{}_scatter_{}.png".format(attr1,attr2,
                                 urlify(structure_type))), bbox_inches='tight')
        plt.close(fig='all')
        