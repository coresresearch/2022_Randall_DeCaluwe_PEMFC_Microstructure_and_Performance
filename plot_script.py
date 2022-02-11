
import os
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from Model.Shared_Funcs.read_and_write import *
from Model import pemfc_runner

plt.close('all')

def fig_starter(fig_num):
    while plt.fignum_exists(fig_num):
        fig_num = fig_num +1
        
    return fig_num

font = {'family':'Arial','size':10}
plt.rc('font',**font)

""" Polarization comparison """
# This figure plots the polarization curves vs. the Owejan data for the model
# results assuming literautre properties, bulk-like, and mixed.

def validate_plot(condition,**kwargs):
    
    name = kwargs.get('name','validation_'+condition)
    
    method = ['lit','bulk','mix']

    x_list,y_list = {},{}
    df_i0,df_p0,df_f0,df_y0,df_r0 = {},{},{},{},{}
    df_i1,df_p1,df_f1,df_y1,df_r1 = {},{},{},{},{}
    df_i2,df_p2,df_f2,df_y2,df_r2 = {},{},{},{},{}
    df_i3,df_p3,df_f3,df_y3,df_r3 = {},{},{},{},{}

    for i,n in enumerate(method):
        load_folder0 = n+'_2_'+condition
        directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'
        df_i0[n],df_p0[n],df_f0[n],df_y0[n],df_r0[n] = loader(directory0)
        x_list[n+'_2'] = df_p0[n]['i_ext [A/cm2]']
        y_list[n+'_2'] = df_p0[n]['Voltage [V]']
        
        load_folder1 = n+'_1_'+condition
        directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'
        df_i1[n],df_p1[n],df_f1[n],df_y1[n],df_r1[n] = loader(directory1)
        x_list[n+'_1'] = df_p1[n]['i_ext [A/cm2]']
        y_list[n+'_1'] = df_p1[n]['Voltage [V]']
        
        load_folder2 = n+'_05_'+condition
        directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'
        df_i2[n],df_p2[n],df_f2[n],df_y2[n],df_r2[n] = loader(directory2)
        x_list[n+'_05'] = df_p2[n]['i_ext [A/cm2]']
        y_list[n+'_05'] = df_p2[n]['Voltage [V]']
        
        load_folder3 = n+'_025_'+condition
        directory3 = os.getcwd()+'/Results/'+load_folder3+'/Saved_dfs'
        df_i3[n],df_p3[n],df_f3[n],df_y3[n],df_r3[n] = loader(directory3)
        x_list[n+'_025'] = df_p3[n]['i_ext [A/cm2]']
        y_list[n+'_025'] = df_p3[n]['Voltage [V]']
    
    # All x values are shared, y* is the y data, and s* is the error bars (+/-)
    if condition == 'air':
        x = np.array([0.0,0.05,0.20,0.40,0.80,1.0,1.2,1.5,1.65,1.85,2.0])
    elif condition == 'o2':
        x = np.array([0.03,0.05,0.10,0.2,0.4,0.8,1.0,1.2,1.5,1.75,2.0])
    
    # w_Pt = 0.2 mg/cm^2
    if condition == 'air':
        y0 = np.array([0.95,0.85,0.80,0.77,0.73,0.72,0.70,0.68,0.67,0.65,0.63]) 
        s0 = np.array([0.1,12,7,7,12,1,8,7,7,9,9]) *1e-3 
    elif condition == 'o2':
        y0 = np.array([0.90,0.89,0.86,0.84,0.81,0.77,0.76,0.75,0.73,0.71,0.70])
        s0 = np.array([5,4,8,5,5,5,5,5,6,9,10]) *1e-3
    
    # w_Pt = 0.1 mg/cm^2
    if condition == 'air':
        y1 = np.array([0.93,0.83,0.79,0.75,0.71,0.69,0.67,0.65,0.64,0.62,0.60]) 
        s1 = np.array([0.1,9,7,5,7,11,11,7,9,11,11]) *1e-3
    elif condition == 'o2':
        y1 = np.array([0.89,0.86,0.84,0.81,0.78,0.74,0.73,0.71,0.69,0.68,0.67])
        s1 = np.array([5,9,5,5,5,5,5,5,8,9,10]) *1e-3
    
    # w_Pt = 0.05 mg/cm^2
    if condition == 'air':
        y2 = np.array([0.92,0.81,0.76,0.72,0.67,0.65,0.63,0.60,0.59,0.56,0.54]) 
        s2 = np.array([0.1,8,6,6,7,7,5,5,6,7,7]) *1e-3
    elif condition == 'o2':
        y2 = np.array([0.86,0.83,0.81,0.78,0.75,0.71,0.69,0.67,0.65,0.63,0.62])
        s2 = np.array([8,8,6,6,7,8,8,8,9,8,7]) *1e-3
    
    # w_Pt = 0.025 mg/cm^2
    if condition == 'air':
        y3 = np.array([0.91,0.79,0.72,0.68,0.63,0.60,0.57,0.53,0.50,0.46,0.43]) 
        s3 = np.array([0.1,4,10,14,13,13,19,24,25,23,24]) *1e-3
    elif condition == 'o2':
        y3 = np.array([0.84,0.81,0.78,0.75,0.72,0.67,0.65,0.64,0.61,0.59,0.57])
        s3 = np.array([6,5,5,5,8,12,13,14,16,18,20]) *1e-3
        
    # color palette
    c = ['C0','C1','C2','C3']
    
    # titles
    t = ['(a) Literature','(b) Uniform','(c) Mixed']
    
    fig_num = fig_starter(0)
    fig = plt.figure(fig_num)
    fig.set_size_inches(7.45,1.75)
    
    for i,n in enumerate(method):
    
        ax1 = plt.subplot(1,3,i+1)
        ax1.set_xlabel(r'Current Density [A/cm$^2$]',labelpad=1)
        
        ax1.set_xlim([0, 2.1])
        ax1.set_ylim([0.4, 1.0])
        
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax1.tick_params(axis='y',which='both',direction='in',right=True)
        ax1.tick_params(axis='x',which='both',direction='in',top=True,pad=5)
        
        ax1.set_xticks([0.5*ind for ind in range(5)])
        
        if i != 0: ax1.set_yticklabels([])
        # if i == 2: ax1.set_xlabel(r'Current Density [A/cm$^2$]',labelpad=2.25)
    
        if i == 2:
            opt_lin = {'color':'black','linewidth':2,'label':'Model'}
            opt_err = {'fmt':'.','color':'black','capsize':3,'label':'Owejan et al.'}
            ax1.plot(x_list[n+'_2'][0],y_list[n+'_2'][0],**opt_lin)
            ax1.errorbar(x[0],y0[0],yerr=s0[0],**opt_err)
    
        ax1.plot(x_list[n+'_2'],y_list[n+'_2'],color=c[0],linewidth=2)
        ax1.errorbar(x,y0,yerr=s0,fmt='.',color=c[0],capsize=3)
        
        ax1.plot(x_list[n+'_1'],y_list[n+'_1'],color=c[1],linewidth=2)
        ax1.errorbar(x,y1,yerr=s1,fmt='.',color=c[1],capsize=3)
        
        ax1.plot(x_list[n+'_05'],y_list[n+'_05'],color=c[2],linewidth=2)
        ax1.errorbar(x,y2,yerr=s2,fmt='.',color=c[2],capsize=3)
         
        ax1.plot(x_list[n+'_025'],y_list[n+'_025'],color=c[3],linewidth=2)
        ax1.errorbar(x,y3,yerr=s3,fmt='.',color=c[3],capsize=3)
        
        # ax1.set_title(t[i],fontsize=12,pad=5)
        
        if i == 2: ax1.legend(loc='lower left',fontsize=10,frameon=False,
                              labelspacing=0.25)
        
        # if i == 1 or i == 2: ax1.set_yticklabels([])
        # if i == 0 or i == 1: ax1.set_xticklabels([])
        else: ax1.set_xticklabels([0.5*ind for ind in range(5)]) 
        
    plt.subplots_adjust(wspace=0.1)
    # plt.subplots_adjust(hspace=0.15)
    
    fig.add_subplot(111,frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel(r'Cell Voltage [V]',labelpad=-5)
    
    plt.savefig('Figures/'+name+'_chk2.png',dpi=500,bbox_inches='tight')
    
# validate_plot('air')
# validate_plot('o2')

""" Losses cause """
# This figure plots the differences between the 0.2 and 0.025 mg/cm2 Pt cases
# taken at 2.0 A/cm2. It is meant to highlight differences and similarities in
# gradients that affect the cell performance.

def losses_cause_plots():

    method = ['mix_2_air','mix_025_air']
    
    load_folder0 = method[0]
    directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'
    df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
    
    load_folder1 = method[1]
    directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'
    df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
    
    x_col = 'Depth [um]'
    y_col = ['eps_w [-]',None,'i_far_frac [-]']
    norm = [0.1, 1, 1]        # divide eps_w by eps_go when needed
    shift = 250               # GDL thickness in microns
    
    # color palette
    c = ['C0','C3']
    
    # title
    t = ['(a)','(b)','(c)']
    
    # gridspec
    # gridspec = {'wspace':0.25,'width_ratios':[1,1,1]}
    
    fig_num = fig_starter(0)
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=[3.07,6])
    
    opt_0 = {'color':c[0],'linewidth':2,'marker':'o'}
    opt_1 = {'color':c[1],'linewidth':2,'marker':'o'}
    
    for i in range(3):
        
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[i].tick_params(axis='x',which='both',direction='in',top=True,pad=5)
        ax[i].tick_params(axis='y',which='both',direction='in',right=True)
        
        ax[i].set_xticks(np.linspace(0,15,7))
        ax[i].set_xticklabels(np.linspace(0,15,7))
        ax[i].set_xlim([0,15])
        
        if i == 0:
            x_dat = df_y0[x_col][10:].to_numpy()-shift
            y_dat1 = df_y0[y_col[i]][10:].to_numpy()/norm[i]
            y_dat2 = df_y1[y_col[i]][10:].to_numpy()/norm[i]
            
            y_dat1[0] = 0.03
            y_dat2[0] = 0.03
            
            ax[i].plot(x_dat,y_dat1,**opt_0)
            ax[i].plot(x_dat,y_dat2,**opt_1)
            
            ax[i].set_ylabel(r'${\cal S}_{\rm w}$ [-]')
            ax[i].set_yticks(np.round(np.linspace(0,0.4,5),1))
            ax[i].set_yticklabels(np.round(np.linspace(0,0.4,5),1))
            
            ax[i].set_ylim([0,0.42])
            
        elif i == 1:
            rho_naf_o2_0 = df_y0['rho_k r2 O2(Naf)'][10:]
            rho_naf_o2_1 = df_y1['rho_k r2 O2(Naf)'][10:]
            
            delta_rho_o2_0 = df_y0['rho_k r0 O2(Naf)'][10:] - rho_naf_o2_0
            delta_rho_o2_1 = df_y1['rho_k r0 O2(Naf)'][10:] - rho_naf_o2_1
            
            opt_0_sp = copy.deepcopy(opt_0)
            opt_0_sp['linestyle'] = '-'
            
            opt_1_sp = copy.deepcopy(opt_1)
            opt_1_sp['linestyle'] = '-'
            
            ax[i].set_ylabel(r'$\rho_{\rm O2, N}$ [kg/m$^3$]')
            ax[i].plot(df_y0[x_col][10:]-shift,rho_naf_o2_0,**opt_0_sp)
            ax[i].plot(df_y1[x_col][10:]-shift,rho_naf_o2_1,**opt_1_sp)
            ax[i].set_yticks(np.linspace(0,10,6).astype(int))
            ax[i].set_yticklabels(np.linspace(0,10,6).astype(int))
            
            # ax2 = ax[i].twinx()   
            # ax2.yaxis.set_minor_locator(AutoMinorLocator())
            # ax2.set_ylabel(r'$\Delta\rho_{\rm O2(Naf)}$ [kg/m$^3$]')
            # ax2.tick_params(axis='y',which='both',direction='in')
            
            # ax2.plot(df_y0[x_col][10:]-shift,delta_rho_o2_0,**opt_0)
            # ax2.plot(df_y1[x_col][10:]-shift,delta_rho_o2_1,**opt_1)
            # ax2.set_yticks(np.round(np.linspace(0,1,6),1))
            # ax2.set_yticklabels(np.round(np.linspace(0,1,6),1))
            
            # box = ax[i].get_position()
            # box.x0 = box.x0 - 0.005
            # box.x1 = box.x1 - 0.005
            # ax[i].set_position(box)
            
        elif i == 2:
            ax[i].plot(df_y0[x_col][10:]-shift,df_y0[y_col[i]][10:]/norm[i],**opt_0)
            ax[i].plot(df_y1[x_col][10:]-shift,df_y1[y_col[i]][10:]/norm[i],**opt_1)
            
            ax[i].set_ylabel(r'$j_{\rm ~Far} \ / \ \bf{j}_{\rm ~ext}$ [-]') 
            ax[i].set_yticks(np.round(np.linspace(0,0.5,6),1))
            ax[i].set_yticklabels(np.round(np.linspace(0,0.5,6),1))
            
            ax[i].set_xlabel(r'Catalyst Layer Depth [$\mu$m]')
            
        if i != 2: ax[i].set_xticklabels([])
        
        # ax[i].set_title(t[i],fontsize=10,pad=10)
        
    plt.subplots_adjust(hspace=0.15)
    
    fig.add_subplot(111,frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
        
    plt.savefig('Figures/causes_of_losses_vert.png',dpi=500,bbox_inches='tight')
    
# losses_cause_plots()

""" theta_pt vs. i """
# For both the high and low Pt loading cases, plot the theta_pt values at each
# current when liquid water is modeled vs not. Only consider the node closest
# to the PEM.

nw_method = ['t12_pt_eql_2_nw_15um_cl','t12_pt_eql_025_nw_15um_cl']
ww_method = ['t12_pt_eql_2_ww_15um_cl','t12_pt_eql_025_ww_15um_cl']

def plot_v_i(*method,**options):
    
    x_col,y_col = 'i_ext [A/cm2]',options['y_col']
    x_label,y_label = r'Current Density [A/cm$^2$]',options['y_label']    
    
    # color palette
    c = ['C0','C3']
    
    # title
    t = ['(a) without liquid phase','(b) with liquid phase']
    
    # line options
    opt_0 = {'color':c[0],'linewidth':2}
    opt_1 = {'color':c[1],'linewidth':2}
    
    # Start figure
    fig_num = fig_starter(0)
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=[10,3])
    
    for i,m in enumerate(method):
        load_folder0 = m[0]
        directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'
        df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
        
        load_folder1 = m[1]
        directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'
        df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1) 
        
        if options['depth']: 
            x_vals0,y_vals0 = df_f0[x_col],df_f0[y_col]
            x_vals1,y_vals1 = df_f1[x_col],df_f1[y_col]
        if options['power']: 
            x_vals0,y_vals0 = df_p0[x_col],df_p0[y_col]
            x_vals1,y_vals1 = df_p1[x_col],df_p1[y_col]
        
        ax[i].plot(x_vals0,y_vals0,**opt_0)
        ax[i].plot(x_vals1,y_vals1,**opt_1)
        
        ax[i].set_xlim(options['x_lim'])
        ax[i].set_ylim(options['y_lim'])
        
        ax[i].tick_params(axis='x',direction='in',top=True)
        ax[i].tick_params(axis='y',direction='in',right=True)
        
        ax[i].set_xlabel(x_label)
        ax[i].set_ylabel(y_label)
        ax[i].set_title(t[i],fontsize=14,pad=10)
        
    legend = [r'0.2',r'0.025']
    ax[1].legend(legend,frameon=False,handlelength=1,loc='best')
        
    plt.subplots_adjust(wspace=0.4)
        
options = {'y_col':'theta_pt_k y4 Pt(S)','y_label':r'$\theta_{\rm Pt(S)}$ [-]',
           'depth':1,'power':0,'x_lim':[0,5],'y_lim':[0.1,1.05]}
# plot_v_i(nw_method,ww_method,**options)

options = {'y_col':'rho_naf_k y4 r2 O2(Naf)','y_label':r'$\rho_{\rm O2(Naf)}$ [kg/m$^3$]',
           'depth':1,'power':0,'x_lim':[0,5],'y_lim':[0,15]}
# plot_v_i(nw_method,ww_method,**options)

options = {'y_col':'rho_gas_k y4 O2','y_label':r'$\rho_{\rm O2(gas)}$ [kg/m$^3$]',
           'depth':1,'power':0,'x_lim':[0,5],'y_lim':[0,0.4]}
# plot_v_i(nw_method,ww_method,**options)

options = {'y_col':'Power [W/cm2]','y_label':r'Power Density [W/cm$^2$]',
           'depth':0,'power':1,'x_lim':[0,5],'y_lim':[0,1.75]}
# plot_v_i(nw_method,ww_method,**options)

def plot_loop_v_i(df_f0,df_y0,df_f1,df_y1):
    
    ind_all = [j for j in df_f0.columns if 'O2' in j]
    ind_act = [k for k in ind_all if not '(Naf)' in k]
    
    pull = {}
    
    # use for 'bulk' runs
    # pull[0] = [0,10,11,13,15,17,18,19,25,28,29]
    # pull[1] = [0,10,11,13,15,16,17,18,20,21,22]
    
    # use for 'mix' runs
    pull[0] = [0,16,22,28,29,34,45]
    pull[1] = pull[0]
    
    df_f,df_y = {},{}
    df_f[0],df_y[0] = df_f0,df_y0
    df_f[1],df_y[1] = df_f1,df_y1
    
    title = ['(a) without liquid phase', '(b) with liquid phase']
    
    spacing = np.linspace(0,1,len(pull[0]))
    c_vec = [plt.cm.seismic(j) for j in spacing]
    
    fig_num = fig_starter(0)
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=[2.4,4])    
    
    for n in range(2):
        nxt = 0
        for i,i_ext in enumerate(df_f[n]['i_ext [A/cm2]']):
            if i in pull[n]:
                y_vals = df_f[n].iloc[i][ind_act].to_numpy()
                ax[n].plot(df_y[n]['Depth [um]'],y_vals,color=c_vec[nxt],linewidth=2)
                nxt = nxt+1
        
        ax[n].set_xlim([0,300])
        ax[n].set_ylim([0.10,0.35])
        
        if n == 0: ax[n].set_xticklabels([])
        if n == 1: ax[n].set_xlabel(r'Cathode Depth [$\mu$m]',labelpad=2)

        # ax[n].set_title(title[n],fontsize=10)
        
        ax[n].xaxis.set_minor_locator(AutoMinorLocator())
        ax[n].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[n].tick_params(axis='x',which='both',direction='in',top=True)
        ax[n].tick_params(axis='y',which='both',direction='in',right=True)
        
    norm = mpl.colors.Normalize(vmin=0,vmax=2.25)
    clrmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.seismic.reversed())
    clrmap.set_array([])
    
    cbaxes = fig.add_axes([0.95, 0.17, 0.03, 0.65])
    
    cbar = fig.colorbar(clrmap, cax=cbaxes, orientation='vertical')
    cbar.set_label(r'Current Density [A/cm$^2$]', fontsize=10, labelpad=1.5)
    cbar.ax.tick_params(axis='y',direction='in')
    cbar.set_ticks([2.25,1.25,0.25])
    cbar.ax.set_yticklabels([0,1,2], fontsize=10)
    cbar.ax.tick_params(pad=2)

    plt.subplots_adjust(hspace=0.15)
    
    fig.add_subplot(111,frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel(r'$\rho_{\rm O2, g}$ [kg/m$^3$]',labelpad=5)
    
    plt.savefig('Figures/O2_grads_comparison_chk.png',dpi=500,bbox_inches='tight')
        
# load_folder0 = 't12_pt_eql_025_nw_15um_cl'
# directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'
        
# load_folder1 = 't12_pt_eql_025_ww_15um_cl'
# directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

# df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
# df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
# plot_loop_v_i(df_f0,df_y0,df_f1,df_y1)

""" Bar charts """
# For the various design tests that are run, figure out the max power and the
# limiting current. Plot each of them in a bar chart.

def bar_charts():
    i_lim_dfs = {}
    p_max_dfs = {}
    
    for i,f in enumerate(os.listdir(os.getcwd()+'/Results/')):
        load_folder0 = f
        directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'
        df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
        
        i_lim_dfs[f] = max(df_p0['i_ext [A/cm2]'])
        p_max_dfs[f] = max(df_p0['Power [W/cm2]'])
        
    # baseline bin
    i_lim_base = np.array([i_lim_dfs['t12_pt_eql_025_ww_15um_cl']])
    p_max_base = np.array([p_max_dfs['t12_pt_eql_025_ww_15um_cl']])
    
    # graded pt bin
    files = ['t12_pt_lin_025_15um_cl','t12_pt_exp_025_15um_cl']
    labels_graded_pt = ['Linear Pt','Exponential Pt']
    
    i_lim_graded_pt = np.zeros(len(files))
    p_max_graded_pt = np.zeros(len(files))
    
    for i,f in enumerate(files):
        i_lim_graded_pt[i] = i_lim_dfs[f]
        p_max_graded_pt[i] = p_max_dfs[f]
    
    # graded nafion bin
    files = ['t_glob_pt_eql_025_15um_cl','t_inc_pt_eql_025_15um_cl']
    labels_graded_naf = ['Random Naf','Linear Naf']
    
    i_lim_graded_naf = np.zeros(len(files))
    p_max_graded_naf = np.zeros(len(files))
    
    for i,f in enumerate(files):
        i_lim_graded_naf[i] = i_lim_dfs[f]
        p_max_graded_naf[i] = p_max_dfs[f]
    
    # nafion thicknesses bin
    files = ['t7_pt_eql_025_15um_cl','t18_pt_eql_025_15um_cl']
    labels_t_nafs = [r'7$~$nm Naf',r'18$~$nm Naf']
    
    i_lim_t_nafs = np.zeros(len(files))
    p_max_t_nafs = np.zeros(len(files))
    
    for i,f in enumerate(files):
        i_lim_t_nafs[i] = i_lim_dfs[f]
        p_max_t_nafs[i] = p_max_dfs[f]
    
    # CL thicknesses bin
    files = ['t12_pt_eql_025_9um_cl','t12_pt_eql_025_12um_cl']
    labels_t_cls = [r'9$~\mu$m CL',r'12$~\mu$m CL']
    
    i_lim_t_cls = np.zeros(len(files))
    p_max_t_cls = np.zeros(len(files))
    
    for i,f in enumerate(files):
        i_lim_t_cls[i] = i_lim_dfs[f]
        p_max_t_cls[i] = p_max_dfs[f]
    
    # dual graded bin
    files = ['t_inc_pt_exp_025_15um_cl']
    labels_dual_grad = ['Dual Graded']
    
    i_lim_dual_grad = np.zeros(len(files))
    p_max_dual_grad = np.zeros(len(files))
    
    for i,f in enumerate(files):
        i_lim_dual_grad[i] = i_lim_dfs[f]
        p_max_dual_grad[i] = p_max_dfs[f]
        
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=[6.45,2.5])
    
    labels = ['Base Case']
    i_lims = i_lim_base
    p_maxs = p_max_base
    c_vec = ['C3']
    
    labels.extend(labels_t_cls)
    i_lims = np.hstack([i_lims,i_lim_t_cls])
    p_maxs = np.hstack([p_maxs,p_max_t_cls])
    c_vec.extend(['orange' for j in range(len(labels_t_cls))])
    
    labels.extend(labels_t_nafs)
    i_lims = np.hstack([i_lims,i_lim_t_nafs])
    p_maxs = np.hstack([p_maxs,p_max_t_nafs])
    c_vec.extend(['green' for j in range(len(labels_t_nafs))])
    
    labels.extend(labels_graded_naf)
    i_lims = np.hstack([i_lims,i_lim_graded_naf])
    p_maxs = np.hstack([p_maxs,p_max_graded_naf])
    c_vec.extend(['blue' for j in range(len(labels_graded_naf))])
    
    labels.extend(labels_graded_pt)
    i_lims = np.hstack([i_lims,i_lim_graded_pt])
    p_maxs = np.hstack([p_maxs,p_max_graded_pt])
    c_vec.extend(['purple' for j in range(len(labels_graded_pt))])
    
    labels.extend(labels_dual_grad)
    i_lims = np.hstack([i_lims,i_lim_dual_grad])
    p_maxs = np.hstack([p_maxs,p_max_dual_grad])
    c_vec.extend(['black' for j in range(len(labels_dual_grad))])
    
    x_pos = np.flip(np.array([0,  1.75,3,  4.75,6,  7.75,9,  10.75,12,  13.75]))
    width = 1.0
    
    ax[0].barh(x_pos,p_maxs,width,color=c_vec)
    ax[1].barh(x_pos,i_lims,width,color=c_vec)    
    
    for n in range(2):
        ax[n].set_yticks(x_pos)
        
        if n == 0: 
            ax[n].set_yticklabels(labels)
            ax[n].set_xlabel(r'Max power density [W/cm$^2$]',fontsize=10)
            ax[n].set_title('(a)',fontsize=10,pad=5)
            ax[n].set_xlim([0.,1.2])
            ax[n].set_ylim([-1,14.75])
            
            ax[n].set_xticks(np.round(np.linspace(0,1.2,7),1))
            ax[n].set_xticklabels(np.round(np.linspace(0,1.2,7),1))
            
            thresh_opts = {'linestyle':'--','linewidth':2,'color':'grey'}
            ax[n].plot([p_max_base,p_max_base],[-1,15],**thresh_opts)
        
        if n == 1: 
            ax[n].set_yticklabels([])
            ax[n].set_xlabel(r'$j_{\rm \ Lim}$ [A/cm$^2$]',fontsize=10)
            ax[n].set_title('(b)',fontsize=10,pad=5)
            ax[n].set_xlim([0.,3.0])
            ax[n].set_ylim([-1,14.75])
            
            ax[n].set_xticks(np.round(np.linspace(0,3,7),1))
            ax[n].set_xticklabels(np.round(np.linspace(0,3,7),1))
            
            thresh_opts = {'linestyle':'--','linewidth':2,'color':'grey'}
            ax[n].plot([i_lim_base,i_lim_base],[-1,15],**thresh_opts)
            
        ax[n].xaxis.set_minor_locator(AutoMinorLocator())
        ax[n].tick_params(axis='x',which='both',direction='in',top=True,pad=5)
        
        ax[n].tick_params(axis='y',direction='in')
        
    plt.subplots_adjust(wspace=0.15)
    plt.savefig('Figures/bar-charts-6_45in.png',dpi=500,bbox_inches='tight')
    
# bar_charts()

""" Sigma vs CL depth for 0.2 and 0.025 mg_pt/cm2 at i = 2.0 A/cm2 """
# To figure out how much of a difference the ionic conductivity of the mixed 
# model makes at high currents between the high and low pt loading cases, plot
# the values at each CL node.

# args = ['mix_2_air','mix_025_air','bulk_025_air']
    
# fig,ax = plt.subplots(nrows=1,ncols=1)

# options = {'linewidth':2,'marker':'o'}

# for i_f,f in enumerate(args):
#     cwd = os.getcwd()
    
#     os.chdir(cwd+'/Results/'+f)
#     print(os.getcwd())
    
#     if i_f == 0: from Results.mix_2_air.pemfc_runner import *
#     elif i_f == 1: from Results.mix_025_air.pemfc_runner import *
#     elif i_f == 2: from Results.bulk_025_air.pemfc_runner import *
    
#     print(w_Pt)
#     exec(open("Shared_Funcs/pemfc_pre.py").read())
#     os.chdir(cwd)
    
#     directory0 = os.getcwd()+'/Results/'+f+'/Saved_dfs'
#     df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
    
#     cl.update(ca,df_i0.iloc[27][1:].to_numpy())
    
#     if i_f == 0: options['color'] = 'C0'
#     elif i_f == 1: options['color'] = 'C3'
#     elif i_f == 2: options['color'] = 'black'
    
#     ax.plot(df_y0['Depth [um]'][10:]-gdl.d['y']*1e6,cl.d['sig_io'],**options)   

# ax.set_xlim([0,15])
# ax.set_ylim([2.5,5.0])

# ax.set_xlabel(r'Catalyst Layer Depth [$\mu$m]')
# ax.set_ylabel(r'$\sigma_{\rm \ io , Naf}$ [S/m]')

# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.yaxis.set_minor_locator(AutoMinorLocator())

# ax.tick_params(axis='x',which='both',direction='in',top=True,pad=8)
# ax.tick_params(axis='y',which='both',direction='in',right=True)

# ax.legend(['0.2','0.025','bulk'],loc='lower right',frameon=False)

# plt.savefig('Figures/sigma_v_y.png',dpi=500,bbox_inches='tight')     

""" Effective ohmic resistance """
# This figure creates a plot for effective ohmic overpotential and/or 
# resistance. eta_eff_ohm is defined as 
#
#       sum_{i=1}^{Ny_cl} i_{io,j} / sigma_j * i_{io,j} / i_ext * t_j.
#
# where i_{io,j} = i_{io,j-1} - i_{far,j} and i_{io,j} = i_ext at j = Ny_cl. 
# R_eff_ohm is then defined as eta_eff_ohm / i_ext.

def ohm_eff_load(method):

    load_folder0 = method+'_2_air'
    directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'
    
    load_folder1 = method+'_1_air'
    directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'
    
    load_folder2 = method+'_05_air'
    directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'
    
    load_folder3 = method+'_025_air'
    directory3 = os.getcwd()+'/Results/'+load_folder3+'/Saved_dfs'
    
    df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
    df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
    df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)
    df_i3,df_p3,df_f3,df_y3,df_r3 = loader(directory3)
    
    args = [method+'_2_air',method+'_1_air',method+'_05_air',method+'_025_air']

    return df_i0,df_i1,df_i2,df_i3,args

def ohm_eff_plot(df_i,cl,ca):
    
    eta_eff = np.zeros(df_i[0].shape[0])
    
    for i_c,i_ext in enumerate(df_i[0][1:].to_numpy()):
        
        sv = df_i.iloc[i_c+1][1:].to_numpy()
        i_ext = i_ext *100**2
        
        cl.update(ca,sv)
        
        t_j = cl.d['dy']
        i_io_jm1 = i_ext
        
        epstau_n = cl.d['eps/tau2_n'][-1]
        sig_eff = cl.d['sig_io'][-1] *epstau_n
        
        for i_y in range(cl.d['Ny']):
            
            j = cl.d['Ny'] - i_y - 1
            
            ca.inner_rxn_state(cl,sv,j)
            
            i_far_j = -ca.pt_s[j].get_net_production_rates(ca.carb)*ct.faraday\
                    *cl.d['SApv_pt'] *cl.d['dy']
            
            i_io_j = i_io_jm1 - i_far_j
            new_term = i_io_j**2 / sig_eff / i_ext * t_j
                        
            eta_eff[i_c+1] = eta_eff[i_c+1] + new_term
            
            i_io_jm1 = i_io_j
            epstau_n = np.mean(cl.d['eps/tau2_n'][j-1:j+1])
            sig_eff = np.mean(cl.d['sig_io'][j-1:j+1]) *epstau_n
            
    return eta_eff
    
# df_i0,df_i1,df_i2,df_i3,args = ohm_eff_load('mix')

# eta_ohm_eff = np.zeros([4,df_i0.shape[0]])
# R_ohm_eff = np.zeros([4,df_i0.shape[0]])

# for i_f,f in enumerate(args):
#     cwd = os.getcwd()
    
#     if i_f == 0: from Results.mix_2_air.pemfc_runner import *
#     elif i_f == 1: from Results.mix_1_air.pemfc_runner import *
#     elif i_f == 2: from Results.mix_05_air.pemfc_runner import *
#     elif i_f == 3: from Results.mix_025_air.pemfc_runner import *
    
#     os.chdir(cwd+'/Results/'+f)
    
#     print(os.getcwd())    
#     print(w_Pt)    
#     exec(open("Shared_Funcs/pemfc_pre.py").read())
    
#     if i_f == 0: df_i = df_i0
#     elif i_f == 1: df_i = df_i1
#     elif i_f == 2: df_i = df_i2
#     elif i_f == 3: df_i = df_i3
    
#     eta_ohm_eff[i_f,:] = ohm_eff_plot(df_i,cl,ca)
#     R_ohm_eff[i_f,1:] = eta_ohm_eff[i_f,1:] / df_i[0][1:].to_numpy()
    
#     os.chdir(cwd)
    
# i_plot = df_i0[0].to_numpy()
    
# fig_num = fig_starter(0)
# fig,ax = plt.subplots(nrows=2,ncols=1,figsize=[3.07,4])

# for i in range(4):    
#     ax[0].plot(i_plot,eta_ohm_eff[i,:],linewidth=2)
#     ax[1].plot(i_plot[1:],R_ohm_eff[i,1:],linewidth=2)
    
# # ax[0].set_title('(a)',pad=10)
# # ax[1].set_title('(b)',pad=10)
    
# # ax[0].set_xlabel(r'Current Density [A/cm$^2$]')
# ax[0].set_ylabel(r'$\eta_{\rm Ohm}^{\rm eff}$ [V]')

# ax[1].set_xlabel(r'Current Density [A/cm$^2$]')
# ax[1].set_ylabel(r'$R_{\rm Ohm}^{\rm eff}$ [$\Omega$-cm$^2$]')

# ax[0].set_xlim([0.,2.1])
# ax[0].set_ylim([0.,0.12])

# ax[0].set_yticks([0.03*cnt for cnt in range(5)])
# ax[0].set_yticklabels(['0.0','0.03','0.06','0.09','0.12'])

# ax[1].set_xlim([0.,2.1])
# ax[1].set_ylim([0.,0.08])

# ax[0].xaxis.set_minor_locator(AutoMinorLocator())
# ax[0].yaxis.set_minor_locator(AutoMinorLocator())

# ax[1].xaxis.set_minor_locator(AutoMinorLocator())
# ax[1].yaxis.set_minor_locator(AutoMinorLocator())

# ax[0].tick_params(axis='x',which='both',direction='in',top=True,pad=5)
# ax[0].tick_params(axis='y',which='both',direction='in',right=True)

# ax[1].tick_params(axis='x',which='both',direction='in',top=True,pad=5)
# ax[1].tick_params(axis='y',which='both',direction='in',right=True)

# ax[0].set_xticklabels([])
        
# plt.subplots_adjust(hspace=0.15)    
# plt.savefig('Figures/effective_ohmic_plots_vert.png',dpi=500,bbox_inches='tight')

""" Supplementary Figures """
# For the supplementary information section, it is useful to have a copy of the
# polarization and power curves for each case considered in the barcharts. The
# generic function below accomplishes this.
def polar_power(*args,**kwargs):
    
    split = kwargs.get('split',None)
    xlim = kwargs.get('xlim',[0,3.0])
    ylim_0 = kwargs.get('ylim_0',[0,1.0])
    ylim_1 = kwargs.get('ylim_1',[0,1.2])
    legend = kwargs.get('legend',None)
    c = kwargs.get('colors',['C'+str(i) for i,df in enumerate(args)])
    
    x_col = 'i_ext [A/cm2]'
    y_col = ['Voltage [V]','Power [W/cm2]']
    
    x_label = r'Current Density [A/cm$^2$]'
    y_label = [r'Voltage [V]',r'Power Density [W/cm$^2$]']
    
    polar_opts = {'linewidth':2}
    power_opts = {'linewidth':2,'linestyle':'--'}
    
    if not split:
        fig_num = fig_starter(0)
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=[6.5,2])
        
        for i,df in enumerate(args):    
            ax[0].plot(df[x_col],df[y_col[0]],color=c[i],**polar_opts)
            ax[1].plot(df[x_col],df[y_col[1]],color=c[i],**power_opts)
            
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel(y_label[0],labelpad=1)
        
        # ax[0].set_title('(a)',fontsize=10,pad=5)
        
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel(y_label[1],labelpad=1)
        
        # ax[1].set_title('(b)',fontsize=10,pad=5)
        
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim_0)
        
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim_1)
        
        ax[0].set_xticks(np.round(np.linspace(0,3,7),1))
        ax[0].set_xticklabels(np.round(np.linspace(0,3,7),1))
        
        ax[1].set_xticks(np.round(np.linspace(0,3,7),1))
        ax[1].set_xticklabels(np.round(np.linspace(0,3,7),1))
        
        ax[0].set_yticks(np.round(np.linspace(0,1,6),1))
        ax[0].set_yticklabels(np.round(np.linspace(0,1,6),1))
        
        ax[1].set_yticks(np.round(np.linspace(0,1.25,6),2))
        ax[1].set_yticklabels(np.round(np.linspace(0,1.25,6),2))
        
        ax[0].xaxis.set_minor_locator(AutoMinorLocator())
        ax[0].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[1].xaxis.set_minor_locator(AutoMinorLocator())
        ax[1].yaxis.set_minor_locator(AutoMinorLocator())
        
        ax[0].tick_params(axis='x',which='both',direction='in',top=True,pad=5)
        ax[0].tick_params(axis='y',which='both',direction='in',right=True)
        
        ax[1].tick_params(axis='x',which='both',direction='in',top=True,pad=5)
        ax[1].tick_params(axis='y',which='both',direction='in',right=True)
        
        ax[0].legend(legend,loc='lower left',frameon=False,
                     handlelength=0.5,borderaxespad=1.0,labelspacing=0.25)
        
        plt.subplots_adjust(wspace=0.35)
            
    else:
        fig_num = fig_starter(0)
        fig,ax1 = plt.subplots(nrows=1,ncols=1,figsize=[2.6,2.0])
            
        ax2 = ax1.twinx()
        
        for i,df in enumerate(args):
            ax1.plot(df[x_col],df[y_col[0]],color=c[i],**polar_opts)
            ax2.plot(df[x_col],df[y_col[1]],color=c[i],**power_opts)
                
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim_0)
        ax2.set_ylim(ylim_1)
        
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label[0],labelpad=3)
        ax2.set_ylabel(y_label[1],labelpad=3)
        
        ax1.set_xticks(np.round(np.linspace(0,3,7),1))
        ax1.set_xticklabels(np.round(np.linspace(0,3,7),1))
        
        ax2.set_yticks(np.round(np.linspace(0,1.2,5),2))
        ax2.set_yticklabels(np.round(np.linspace(0,1.2,5),2))
        
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax1.tick_params(axis='x',which='both',direction='in',top=True,pad=5)
        ax1.tick_params(axis='y',which='both',direction='in')
        ax2.tick_params(axis='y',which='both',direction='in')
        
        ax1.legend(legend,loc='lower center',frameon=False,handlelength=0.5,
                   labelspacing=0.2,borderaxespad=0.1)
    
    plt.savefig('Figures/'+kwargs['name']+'_chk.png',dpi=500,bbox_inches='tight')

# Polarization curves and power density curves for different CL design studies.
# Considering different types of Pt distributions.
load_folder0 = 't12_pt_eql_025_ww_15um_cl'
directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'

load_folder1 = 't12_pt_lin_025_15um_cl'
directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

load_folder2 = 't12_pt_exp_025_15um_cl'
directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'

# df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
# df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
# df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)

# clrs = [plt.cm.Purples(j) for j in [0.33,0.66,0.95]]

# options = {'name':'pt-distributions-6_47in','xlim':[0,2.5],'ylim_1':[0,1.2],
#             'colors':clrs,'legend':['Equal Pt','Linear','Exponential']}
# polar_power(df_p0,df_p1,df_p2,**options)

# Polarization curves and power density curves for different CL design studies.
# Considerating different types of Nafion distributions.
load_folder0 = 't12_pt_eql_025_ww_15um_cl'
directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'

load_folder1 = 't_glob_pt_eql_025_15um_cl'
directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

load_folder2 = 't_inc_pt_eql_025_15um_cl'
directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'

# df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
# df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
# df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)

# clrs = [plt.cm.Blues(j) for j in [0.33,0.66,0.95]]

# options = {'name':'naf-distributions-6_47in','colors':clrs,'ylim_1':[0,1.2],
#             'legend':['Equal Naf','Random','Increasing']}
# polar_power(df_p0,df_p1,df_p2,**options)

# Polarization curves and power density curves for different CL design studies.
# Considering different Nafion thicknesses.
load_folder0 = 't7_pt_eql_025_15um_cl'
directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'

load_folder1 = 't12_pt_eql_025_ww_15um_cl'
directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

load_folder2 = 't18_pt_eql_025_15um_cl'
directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'

# df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
# df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
# df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)

# clrs = [plt.cm.Greens(j) for j in [0.33,0.66,0.95]]

# options = {'name':'naf-thicknesses-6_47in','legend':[r'7$~$nm',r'12$~$nm',r'18$~$nm',
#             r'24$~$nm'],'colors':clrs}
# polar_power(df_p0,df_p1,df_p2,**options)

# Polarization curves and power density curves for different CL design studies.
# Considering different CL thicknesses.
load_folder0 = 't12_pt_eql_025_9um_cl'
directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'

load_folder1 = 't12_pt_eql_025_12um_cl'
directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

load_folder2 = 't12_pt_eql_025_ww_15um_cl'
directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'

# df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
# df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
# df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)

# clrs = [plt.cm.Oranges(j) for j in [0.33,0.66,0.95]]

# options = {'name':'CL-thicknesses-6_47in','legend':[r'9$~\mu$m',r'12$~\mu$m',r'15$~\mu$m'],
#             'colors':clrs}
# polar_power(df_p0,df_p1,df_p2,**options)

# Polarization curves and power density curves for different CL design studies.
# 0.025 baseline vs. graded naf only and dual graded.
load_folder0 = 't12_pt_eql_025_ww_15um_cl'
directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'

load_folder1 = 't_inc_pt_eql_025_15um_cl'
directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

load_folder2 = 't_inc_pt_exp_025_15um_cl'
directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'

df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)

# options = {'name':'dual_grad-6_47','split':None,'colors':['C3','blue','black'],
#             'legend':['Base Case','Linear Naf','Dual Graded']}
# polar_power(df_p0,df_p1,df_p2,**options)

options = {'name':'dual-grad-3_07','split':'no','colors':['C3','blue','black'],
            'legend':['Base Case','Linear Naf','Dual Graded']}
polar_power(df_p0,df_p1,df_p2,**options)

# Polarization curves and power density curves for different CL design studies.
# Considerating different types of Nafion distributions.
load_folder0 = 't12_pt_eql_025_ww_15um_cl'
directory0 = os.getcwd()+'/Results/'+load_folder0+'/Saved_dfs'

load_folder1 = 't18_pt_eql_025_15um_cl'
directory1 = os.getcwd()+'/Results/'+load_folder1+'/Saved_dfs'

load_folder2 = 't_inc_pt_eql_025_15um_cl'
directory2 = os.getcwd()+'/Results/'+load_folder2+'/Saved_dfs'

# df_i0,df_p0,df_f0,df_y0,df_r0 = loader(directory0)
# df_i1,df_p1,df_f1,df_y1,df_r1 = loader(directory1)
# df_i2,df_p2,df_f2,df_y2,df_r2 = loader(directory2)

# options = {'name':'naf_distributions','colors':['C3','C0','C2'],'ylim_1':[0,1.0],
#             'legend':['Uniform 12nm','Uniform 18nm','Increasing']}
# polar_power(df_p0,df_p1,df_p2,**options)