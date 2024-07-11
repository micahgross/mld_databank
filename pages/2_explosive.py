# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:14:54 2024

@author: micah
"""
import streamlit as st
import pandas as pd
import numpy as np
# import os
# import json
from io import BytesIO
import base64
# import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

from streamlit_vertical_slider import vertical_slider
#%%
def filter_signal(signal, sfreq, **kwargs):
    import pandas as pd
    from scipy.signal import butter, filtfilt
    low_pass = kwargs.get('low_pass')# low_pass = 20
    high_pass = kwargs.get('high_pass')# high_pass = 20
    band_pass = kwargs.get('band_pass')# band_pass = [10,450]
    order = kwargs.get('filter_order',3)# order = 3
    if 'band_pass' in kwargs:
        band_pass = [x/(sfreq/2) for x in band_pass]
        b1, a1 = butter(order, band_pass, btype='bandpass')
        filt_sig = filtfilt(b1, a1, signal)
    if 'low_pass' in kwargs:
        low_pass = low_pass/sfreq
        b2, a2 = butter(order, low_pass, btype='lowpass')
        filt_sig = filtfilt(b2, a2, signal)
    if 'high_pass' in kwargs:
        high_pass = high_pass/sfreq
        b2, a2 = butter(order, high_pass, btype='highpass')
        filt_sig = filtfilt(b2, a2, signal)
    if type(signal) == pd.Series:
        filt_sig = pd.Series(filt_sig, index=signal.index)
    return filt_sig

def get_start_by_threshold(xt, info, v_threshold = 0.05):
    to_vmax = xt.loc[:xt['velocity'].idxmax()]# plt.plot(xt['Timestamp'], xt['velocity'])
    end_idx = xt[(xt['velocity']>0) & (xt['Position']<=info['Startposition'])].index.max()# plt.plot(xt['Timestamp'], xt['Position'])
    last_subthreshold_idx = to_vmax[to_vmax['velocity']<v_threshold].index.max()
    if not np.isnan(last_subthreshold_idx):
        thresh_to_xmax = xt.loc[last_subthreshold_idx:end_idx]
    else:# if np.isnan(last_subthreshold_idx)
        last_subthreshold_idx = xt[xt['Timestamp']==info['V Zero Timestamp']].index[0]
        thresh_to_xmax = xt.loc[last_subthreshold_idx:end_idx]
    return thresh_to_xmax
    
def post_plots(figs, pars=['xt', 'vt'], expand=[False, True]):
    title_keys = {
        'xt': 'position-time',
        'vt': 'velocity-time',
        'at': 'acceleration-time',
        'Ft': 'force-time',
        'Pt': 'power-time',
        }
    if len(expand) != len(pars):
        expand = [False for par in pars]
    sorted_exercises = [reps[list(reps.keys())[0]]['exercise']] + list(set([reps[rep_nr]['exercise'] for rep_nr in reps if reps[rep_nr]['exercise'] != reps[list(reps.keys())[0]]['exercise']]))
    for i,par in enumerate(pars):# break
        with st.expander(par+' plots', expanded=expand[i]):
            for exercise in sorted_exercises:#list(set([reps[rep_nr]['exercise'] for rep_nr in reps])):# break
                with st.container():
                    st.divider()
                    st.subheader(exercise)
                    st.plotly_chart(figs[exercise][par], use_container_width=True)
                        
@st.cache_data
def filter_reps(reps, selected_reps):
    reps_filtered = {}
    for rep_nr in reps:
        for exercise in selected_reps:
            if rep_nr in selected_reps[exercise]:
                if selected_reps[exercise][rep_nr]:# == True
                    reps_filtered[rep_nr] = reps[rep_nr].copy()
    return reps_filtered

@st.cache_data
def get_rep_kinematics(upload_file):
    df = pd.read_csv(upload_file, sep='\t', header=None)
    df.columns = df.iloc[0]
    date_idx = df[df.loc[:,'date']=='date'].index
    timestamp_idx = df[df.iloc[:,0]=='Timestamp'].index
    date = datetime.strptime(df.loc[date_idx.min()+1, 'date'], '%Y.%m.%d').date()
    reps = {}
    # figs = {}
    rep_nr = 0
    for i in reversed(date_idx):# break# i = date_idx[-1]
        rep_date = datetime.strptime(df.loc[i+1, 'date'], '%Y.%m.%d').date()
        if rep_date == date:
            rep_nr += 1
            rep_time = datetime.strptime(df.loc[i+1, 'time'], '%H:%M:%S').time()
            j = min([j for j in timestamp_idx if j>i])
            info = {
                par: float(df.loc[i+1, par]) for par in [col for col in df.columns if any([string in col.lower() for string in ['id', 'weight', 'load', 'position', 'timestamp']])]
                }
            info['Athlet'] = df.loc[i+1, 'Athlet']
            info['rep_date'] = str(rep_date)
            info['rep_time'] = str(rep_time)
            xt = df.loc[j+1:].iloc[:,:2].rename(columns={col: df.loc[j,col] for col in df.iloc[:,:2]}).astype(float)# plt.plot(xt['Timestamp'], xt['Position'])
            df = df.drop(df.loc[i:].index)
            if info['V Zero Timestamp'] >= xt['Timestamp'].max() or len(xt[((xt['Position']>0.9*info['Startposition']) & (xt['Timestamp'] > info['Deep Timestamp']))]) == 0:
                continue
            
            xt['velocity'] = np.gradient(
                filter_signal(
                    xt['Position'],
                    1000,
                    low_pass=10
                    )
                )
            xt['acceleration'] = 1000 * np.gradient(xt['velocity'])
            xt_conc = get_start_by_threshold(xt, info)
            xt_prop = xt_conc[xt_conc['acceleration'] > 0]
            if all([len(x)==0 for x in [xt_conc, xt_prop]]):
                continue
            reps[rep_nr] = {**info}# {}
            reps[rep_nr]['rep_datetime'] = str(
                datetime.combine(rep_date, rep_time)
                )
            if info['Startposition'] > 800:
                reps[rep_nr]['exercise'] = 'press'
            else:
                reps[rep_nr]['exercise'] = 'pull'

            reps[rep_nr]['xt_full'] = xt
            reps[rep_nr]['xt_conc'] = xt_conc
            reps[rep_nr]['xt_prop'] = xt_prop
    return reps

@st.cache_data
def get_rep_kinetics(reps, rep_loads):
    for rep_nr in reps:# break
        reps[rep_nr]['Load'] = rep_loads[rep_nr]
        for xt in ['xt_full', 'xt_conc', 'xt_prop']:# break
            Ft = xt.replace('x','F')
            reps[rep_nr][Ft] = pd.DataFrame(
                {
                    'Timestamp': reps[rep_nr][xt]['Timestamp'].values,
                    'net_Force': rep_loads[rep_nr] * reps[rep_nr][xt]['acceleration'],
                    'Force': rep_loads[rep_nr] * (reps[rep_nr][xt]['acceleration'] + 9.81)
                    # 'Power': reps[rep_nr][xt]['Force'] * reps[rep_nr][xt]['velocity']
                    }
                )
            reps[rep_nr][Ft]['Power'] = reps[rep_nr][Ft]['Force'] * reps[rep_nr][xt]['velocity']
    return reps

@st.cache_data
def get_figs(reps):
    figs = {}
    par_keys = {
        'xt': 'Position (mm)',
        'vt': 'velocity (m/s)',
        'at': 'acceleration (m/s/s)',
        'Ft': 'Force (N)',
        'Pt': 'Power (W)',
        }
    for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps])):# break
        exer_reps = [rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise]
        figs[exercise] = {}
        for par in ['xt','vt','at','Ft','Pt']:# break
            k1 = 'xt' if par in ['xt','vt','at'] else 'Ft'
            if not k1+'_full' in reps[list(reps.keys())[0]]:
                continue
            k2 = par_keys[par].split(' ')[0]
            fig = make_subplots(
                rows=1, cols=len(exer_reps),
                subplot_titles=['rep '+str(rep_nr) for rep_nr in exer_reps]
                )
            ymins = []
            ymaxs = []
            for r,rep_nr in enumerate(exer_reps,start=1):# break
                fig.add_trace(# figs[exercise][par]
                    go.Line(
                        x=reps[rep_nr][k1+'_full']['Timestamp'],
                        y=reps[rep_nr][k1+'_full'][k2],
                        # name=k2.lower(),
                        showlegend=False,
                        line=dict(color='grey', dash='dash')
                        ),
                    row=1, col=r
                        )
                fig.add_trace(
                    go.Line(
                        x=reps[rep_nr][k1+'_conc']['Timestamp'],
                        y=reps[rep_nr][k1+'_conc'][k2],
                        name='concentric',#k2.lower(),
                        showlegend=True if r==1 else False,
                        line=dict(color='orange', width=8)
                        ),
                    row=1, col=r
                        )
                fig.add_trace(
                    go.Line(
                        x=reps[rep_nr][k1+'_prop']['Timestamp'],
                        y=reps[rep_nr][k1+'_prop'][k2],
                        name='propulsive',
                        showlegend=True if r==1 else False,
                        line=dict(color='red', width=8)
                        ),
                    row=1, col=r
                        )
                tix = [reps[rep_nr][k1+'_conc']['Timestamp'].min(), reps[rep_nr][k1+'_prop']['Timestamp'].max(), reps[rep_nr][k1+'_conc']['Timestamp'].max()]
                fig.update_xaxes(
                    title_text='time (ms)',
                    range=[
                        reps[rep_nr][k1+'_conc']['Timestamp'].min() - 200,
                        reps[rep_nr][k1+'_full']['Timestamp'].max()
                        ],
                    tickvals=tix,
                    ticktext=[t-min(tix) for t in tix],
                    tickangle=70,
                    row=1, col=r
                    )
                ymins.append(reps[rep_nr][k1+'_full'][k2].min())
                ymaxs.append(reps[rep_nr][k1+'_full'][k2].max())
            for r,rep_nr in enumerate(exer_reps,start=1):# break
                fig.update_yaxes(
                    range=[
                        min(ymins),
                        1.1*max(ymaxs)
                        ],
                    showticklabels=True if r==1 else False,
                    row=1, col=r
                    )
            fig.update_yaxes(
                title_text=par_keys[par],
                row=1, col=1
                )
            # fig.show()
            figs[exercise][par] = fig
    return figs

@st.cache_data
def get_profile_figs(reps_filtered):#, get='Fv'):
    # if not type(get) == list:
    #     get = [get]
        
    Fv_figs = {}
    best_reps = {}
    for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps])): 
        best_reps[exercise] = {}
        Fv_figs[exercise] = px.scatter(
            data_frame=pd.DataFrame(
                # [reps[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in reps if selected_reps[rep_nr]]
                # [st.session_state.reps_filtered[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
                # [st.session_state.reps_filtered[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
                {
                    'velocity': [reps_filtered[rep_nr]['xt_prop']['velocity'].mean() for rep_nr in [rep_nr for rep_nr in reps_filtered if reps_filtered[rep_nr]['exercise']==exercise]],
                    'Force': [reps_filtered[rep_nr]['Ft_prop']['Force'].mean() for rep_nr in [rep_nr for rep_nr in reps_filtered if reps_filtered[rep_nr]['exercise']==exercise]]
                    }
                ),
            x='velocity', y='Force',
            # color='orange'
            )
        exer_reps = [rep_nr for rep_nr in reps_filtered if reps[rep_nr]['exercise']==exercise]
        load_pars = {
            load: {rep_nr: reps_filtered[rep_nr]['Ft_prop']['Force'].mean() for rep_nr in exer_reps if reps_filtered[rep_nr]['Load']==load} for load in list(set([reps_filtered[rep_nr]['Load'] for rep_nr in exer_reps]))
            }
        load_bests = []
        for load in load_pars:# break
            load_bests.append(*[rep_nr for rep_nr in load_pars[load] if load_pars[load][rep_nr]==max([load_pars[load][r] for r in load_pars[load]])])

        for rep_nr in exer_reps:
            Fv_figs[exercise].add_annotation(
                x=reps_filtered[rep_nr]['xt_prop']['velocity'].mean(),
                y=reps_filtered[rep_nr]['Ft_prop']['Force'].mean(),
                text=rep_nr,
                bgcolor='orange'
                )
            if rep_nr in load_bests:
                Fv_figs[exercise].add_scatter(
                    x=[reps_filtered[rep_nr]['xt_prop']['velocity'].mean()],
                    y=[reps_filtered[rep_nr]['Ft_prop']['Force'].mean()],
                    marker=dict(color='red', size=20)
                    )
        best_reps[exercise]['prop_force'] = load_bests
    Pv_figs = {}
    for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps])): 
        Pv_figs[exercise] = px.scatter(
            data_frame=pd.DataFrame(
                # [reps[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in reps if selected_reps[rep_nr]]
                # [st.session_state.reps_filtered[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
                # [st.session_state.reps_filtered[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
                {
                    'velocity': [reps_filtered[rep_nr]['xt_prop']['velocity'].mean() for rep_nr in [rep_nr for rep_nr in reps_filtered if reps_filtered[rep_nr]['exercise']==exercise]],
                    'Power': [reps_filtered[rep_nr]['Ft_prop']['Power'].mean() for rep_nr in [rep_nr for rep_nr in reps_filtered if reps_filtered[rep_nr]['exercise']==exercise]]
                    }
                ),
            x='velocity', y='Power',
            # color='orange'
            )
        exer_reps = [rep_nr for rep_nr in reps_filtered if reps[rep_nr]['exercise']==exercise]
        load_pars = {
            load: {rep_nr: reps_filtered[rep_nr]['Ft_prop']['Power'].mean() for rep_nr in exer_reps if reps_filtered[rep_nr]['Load']==load} for load in list(set([reps_filtered[rep_nr]['Load'] for rep_nr in exer_reps]))
            }
        load_bests = []
        for load in load_pars:# break
            load_bests.append(*[rep_nr for rep_nr in load_pars[load] if load_pars[load][rep_nr]==max([load_pars[load][r] for r in load_pars[load]])])

        for rep_nr in [rep_nr for rep_nr in reps_filtered if reps[rep_nr]['exercise']==exercise]:
            Pv_figs[exercise].add_annotation(
                x=reps_filtered[rep_nr]['xt_prop']['velocity'].mean(),
                y=reps_filtered[rep_nr]['Ft_prop']['Power'].mean(),
                text=rep_nr,
                bgcolor='orange'
                )
            if rep_nr in load_bests:
                Pv_figs[exercise].add_scatter(
                    x=[reps_filtered[rep_nr]['xt_prop']['velocity'].mean()],
                    y=[reps_filtered[rep_nr]['Ft_prop']['Power'].mean()],
                    marker=dict(color='red', size=20),
                    name='best propulsive power for load',
                    showlegend=True if load_bests.index(rep_nr)==0 else False
                    )
        best_reps[exercise]['prop_power'] = load_bests
    return Fv_figs, Pv_figs, best_reps

def generate_excel(df, download_filename, sheet_name, link_text = 'Download Excel'):
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    output = BytesIO()
    with pd.ExcelWriter(output, date_format='dd.mm.yyyy') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        # try:
        #     for col in df.columns:
        #         col_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
        #         col_idx = df.columns.get_loc(col)
        #         writer.sheets['abs'].set_column(col_idx, col_idx, col_length)
        #         writer.sheets['rel'].set_column(col_idx, col_idx, col_length)
        # except:
        #     pass
        writer.close()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    link_text = 'Download Excel'
    if not download_filename.endswith('.xlsx'):
        download_filename = download_filename+'.xlsx'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

#%%
st.set_page_config(layout="wide")
st.write("""

# Web App Upper Body Test on Lifter (explosive)

""")
if 'loads_confirmed' not in st.session_state:
    st.session_state.loads_confirmed = False
st.sidebar.write('')
loads_confirmed = False
reps_selected = False
cols = st.columns(2)
with cols[0]:
    upload_file = st.file_uploader(
        "upload .isom or .csv file",
        type=['csv', 'isom'],
        accept_multiple_files=False,
        )
if upload_file is not None:
    # upload_file = 'D:\\explo-1.csv'
    # upload_file = 'D:\\explo-annatina.isom'
    # upload_file = 'D:\\explo-camille.isom'
    reps = get_rep_kinematics(upload_file)#, figs
    rep_loads = {rep_nr: reps[rep_nr]['Load'] for rep_nr in reps}
    if 'reps' not in st.session_state:
        st.session_state.reps = {**reps}

    with st.expander('adjust rep loads (kg)', expanded=True):
        with st.form(key='rep_loads'):
            confirmed_rep_loads = {}
            cols = st.columns(len(list(set([reps[rep_nr]['exercise'] for rep_nr in reps]))))
            for c,exercise in enumerate([reps[list(reps.keys())[0]]['exercise']] + list(set([reps[rep_nr]['exercise'] for rep_nr in reps if reps[rep_nr]['exercise'] != reps[list(reps.keys())[0]]['exercise']]))):
                exer_reps = [rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise]
                with cols[c]:
                    st.subheader(exercise)
                    subcols = st.columns(
                        [0.08 for r in exer_reps] + [round(1-0.08*len(exer_reps),1)]
                        )
                    for r,rep_nr in enumerate(exer_reps):#[rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise]:
                        # confirmed_rep_loads[rep_nr] = subcols[r].slider(#st.slider(
                        #     'rep '+str(rep_nr),#+' mass (kg)',
                        #     min_value=5,
                        #     max_value=100,
                        #     value=int(reps[rep_nr]['Load']),
                        #     step=1,
                        #     )
                        with subcols[r]:
                            confirmed_rep_loads[rep_nr] = vertical_slider(
                                'rep '+str(rep_nr),
                                min_value=5,
                                max_value=100,
                                default_value=int(reps[rep_nr]['Load']),
                                step=1,
                                value_always_visible=True
                                ) 
            confirm_loads = st.form_submit_button(label='confirm loads')
    if confirm_loads:
        # st.write({rep_nr: st.session_state.reps[rep_nr].keys() for rep_nr in st.session_state.reps})
        st.session_state.loads_confirmed = True
        st.session_state.reps = get_rep_kinetics(# reps = get_rep_kinetics(reps, rep_loads={1:10.0, 2:10.0, 3:20.0, 4:20.0, 5:30.0, 6:30.0})
            st.session_state.reps,
            confirmed_rep_loads
            )
        # st.write({rep_nr: st.session_state.reps[rep_nr].keys() for rep_nr in st.session_state.reps})
        st.session_state.figs = get_figs(st.session_state.reps)# figs = get_figs(reps)
        # post_plots(st.session_state.figs, pars=['xt','vt','Ft'], expand=[False,True,False])
    if st.session_state.loads_confirmed:
        if 'reps_filtered' not in st.session_state:
            st.session_state.reps_filtered = {**st.session_state.reps}# reps_filtered = {**reps}
        if 'Fv_figs' not in st.session_state:
            st.session_state.Fv_figs, st.session_state.Pv_figs, st.session_state.best_reps = get_profile_figs(st.session_state.reps_filtered)
        if 'all_reps_selected' not in st.session_state:
            st.session_state.all_reps_selected = False
        with st.form('selected reps', clear_on_submit=True):
            st.subheader('select / de-select reps')
            selected_reps = {}
            sorted_exercises = [reps[list(reps.keys())[0]]['exercise']] + list(set([reps[rep_nr]['exercise'] for rep_nr in reps if reps[rep_nr]['exercise'] != reps[list(reps.keys())[0]]['exercise']]))
            for exercise in sorted_exercises:# list(set([reps[rep_nr]['exercise'] for rep_nr in reps])):# break
                with st.container():
                    st.divider()
                    st.subheader(exercise)
                    st.plotly_chart(st.session_state.figs[exercise]['vt'], use_container_width=True)
                    st.plotly_chart(st.session_state.figs[exercise]['Ft'], use_container_width=True)
                    st.plotly_chart(st.session_state.figs[exercise]['Pt'], use_container_width=True)
                    rep_nrs = sorted([rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise])
                    cols = st.columns(len(rep_nrs))
                    for c,rep_nr in enumerate(rep_nrs):
                        cols[c].write('load: '+str(round(st.session_state.reps[rep_nr]['Load']))+' kg')
                        cols[c].write('Pmax: '+str(round(st.session_state.reps[rep_nr]['Ft_conc']['Power'].max()))+' W')
                    selected_reps[exercise] = {
                        rep_nr: cols[c].checkbox(
                            'rep_'+str(rep_nr),
                            True if st.session_state.all_reps_selected or rep_nr in st.session_state.best_reps[exercise]['prop_power'] else False,
                            # True if st.session_state.all_reps_selected else (
                            #     True if 'selected_reps' in st.session_state and rep_nr in st.session_state.selected_reps else (
                            #         True if rep_nr in st.session_state.best_reps[exercise]['prop_power'] else False
                            #         )
                            #     ),
                            key='rep_'+str(rep_nr)) for c,rep_nr in enumerate(rep_nrs)
                        }  
            cols = st.columns((0.2, 0.2, 0.8))
            def check_selected():
                st.session_state.all_reps_selected = False
            submit_rep_selection = cols[0].form_submit_button('confirm rep selection', on_click=check_selected)
            def check_all():
                st.session_state.all_reps_selected = True
            select_all_reps = cols[1].form_submit_button('select all reps', on_click=check_all)
            def check_best():
                st.session_state.all_reps_selected = False
            select_best_reps = cols[2].form_submit_button('select best by load', on_click=check_best)

        if submit_rep_selection:
            st.session_state.selected_reps = selected_reps
            st.session_state.reps_filtered = filter_reps(st.session_state.reps, selected_reps)
            st.session_state.Fv_figs, st.session_state.Pv_figs, _ = get_profile_figs(st.session_state.reps_filtered)
        if select_all_reps:
            selected_reps = {exercise: {rep_nr: True for rep_nr in[rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise]} for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps]))}
            st.session_state.selected_reps = selected_reps
            st.session_state.reps_filtered = filter_reps(st.session_state.reps, selected_reps)
            st.session_state.Fv_figs, st.session_state.Pv_figs, _ = get_profile_figs(st.session_state.reps_filtered)
        if select_best_reps:
            selected_reps = {exercise: {rep_nr: True if rep_nr in st.session_state.best_reps[exercise]['prop_power'] else False for rep_nr in [rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise]} for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps]))}
            st.session_state.selected_reps = selected_reps
            st.session_state.reps_filtered = filter_reps(st.session_state.reps, selected_reps)
            st.session_state.Fv_figs, st.session_state.Pv_figs, _ = get_profile_figs(st.session_state.reps_filtered)
        cols = st.columns(len(st.session_state.Pv_figs))
        for c,exercise in enumerate(st.session_state.Pv_figs):
            cols[c].write(exercise)
            cols[c].plotly_chart(st.session_state.Pv_figs[exercise], use_container_width=True)
    
    
        st.header('results')
        data_frame = pd.DataFrame(
            {
                'rep nr': [rep_nr for rep_nr in st.session_state.reps_filtered],
                'date': [reps[rep_nr]['rep_date'] for rep_nr in st.session_state.reps_filtered],
                'time': [reps[rep_nr]['rep_time'] for rep_nr in st.session_state.reps_filtered],
                'exercise': [st.session_state.reps_filtered[rep_nr]['exercise'] for rep_nr in st.session_state.reps_filtered],
                'exercise_rep_nr': [rep_nr for rep_nr in st.session_state.reps_filtered],
                'load (kg)': [st.session_state.reps_filtered[rep_nr]['Load'] for rep_nr in st.session_state.reps_filtered],
    
                'conc_time (ms)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Timestamp'].diff().sum() for rep_nr in st.session_state.reps_filtered],
                'prop_time (ms)': [st.session_state.reps_filtered[rep_nr]['xt_prop']['Timestamp'].diff().sum() for rep_nr in st.session_state.reps_filtered],
                'conc_dist (mm)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Position'].diff().sum() for rep_nr in st.session_state.reps_filtered],
                'prop_dist (mm)': [st.session_state.reps_filtered[rep_nr]['xt_prop']['Position'].diff().sum() for rep_nr in st.session_state.reps_filtered],
                'conc_vel (m/s)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['velocity'].mean() for rep_nr in st.session_state.reps_filtered],
                'prop_vel (m/s)': [st.session_state.reps_filtered[rep_nr]['xt_prop']['velocity'].mean() for rep_nr in st.session_state.reps_filtered],
                'peak_vel (m/s)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['velocity'].max() for rep_nr in st.session_state.reps_filtered],
    
    
                'conc_force (N)': [st.session_state.reps_filtered[rep_nr]['Ft_conc']['Force'].mean() for rep_nr in st.session_state.reps_filtered],
                'prop_force (N)': [st.session_state.reps_filtered[rep_nr]['Ft_prop']['Force'].mean() for rep_nr in st.session_state.reps_filtered],
                'peak_force (N)': [st.session_state.reps_filtered[rep_nr]['Ft_conc']['Force'].max() for rep_nr in st.session_state.reps_filtered],
                'conc_power (W)': [st.session_state.reps_filtered[rep_nr]['Ft_conc']['Power'].mean() for rep_nr in st.session_state.reps_filtered],
                'prop_power (W)': [st.session_state.reps_filtered[rep_nr]['Ft_prop']['Power'].mean() for rep_nr in st.session_state.reps_filtered],
                'peak_power (W)': [st.session_state.reps_filtered[rep_nr]['Ft_conc']['Power'].max() for rep_nr in st.session_state.reps_filtered],
                },
            index=range(len([rep_nr for rep_nr in st.session_state.reps_filtered]))
            )
        st.write(data_frame)
        
        cols = st.columns((0.2,0.8))
        athlete_name = cols[0].text_input('athlete name', key='athlete_name')
        with cols[1]:
            st.write(upload_file.name)
            st.markdown(
                generate_excel(data_frame, ''.join([name.capitalize() for name in reversed(st.session_state.athlete_name.split(' '))])+'_'+reps[list(reps.keys())[0]]['rep_date']+'_explo', 'explo'),
                unsafe_allow_html=True
                )#, sign_digits=3
# st.sidebar.write('')
# # def update_load_masses():
    

# # cols = st.columns(2)
# # if 'lap_key' not in st.session_state:
# #     st.session_state.lap_key = None
# # if 'rep_loads' not in st.session_state:
# #     st.session_state.rep_loads = {}
# body_mass = np.nan
# data = None
# loads_confirmed = False
# reps_selected = False
# cols = st.columns(2)
# with cols[0]:
#     upload_file = st.file_uploader(# cols[0].file_uploader(
#         # "upload .fit file",
#         # type=['fit'],#, 'txt'],
#         # accept_multiple_files=False,
#         "upload .csv file",
#         type=['csv', 'isom'],
#         accept_multiple_files=False,
#         )
# if upload_file is not None:# and len(upload_files) != 0
#     # upload_file = [f for f in upload_files if 'ex' in f.name][0]
#     # upload_file = 'C:\\Users\\user\\OneDrive\\python_scripts\\UpperBody_Lifter\\sample_data\\2024-01-26_camille\\camille_explo_24.isom'
#     # upload_file = 'C:\\Users\\user\\OneDrive\\python_scripts\\UpperBody_Lifter\\sample_data\\2024-01-25_mg\\mg-expl-240125.isom'
#     # upload_file = 'D:\\explo-simon.isom'
#     # upload_file = 'D:\\explo-annatina.isom'
    
#     st.write(upload_file.name)
#     #%%
#     @st.cache_data
#     def get_rep_kinematics(upload_file):
#         print('running get_rep_kinematics()')
#         print(datetime.now().time())
#         df = pd.read_csv(upload_file, sep='\t', header=None)
#         df.columns = df.iloc[0]
#         date_idx = df[df.loc[:,'date']=='date'].index
#         timestamp_idx = df[df.iloc[:,0]=='Timestamp'].index
#         date = datetime.strptime(df.loc[date_idx.min()+1, 'date'], '%Y.%m.%d').date()
#         # date_str = str(date)
#         reps = {}
#         figs = {}
#         rep_nr = 0
#         for i in reversed(date_idx):# break# i = date_idx[-1]
#             # print(rep_nr+1, i)
#             rep_date = datetime.strptime(df.loc[i+1, 'date'], '%Y.%m.%d').date()
#             if rep_date == date:
#                 rep_nr += 1
#                 rep_time = datetime.strptime(df.loc[i+1, 'time'], '%H:%M:%S').time()
#                 j = min([j for j in timestamp_idx if j>i])
#                 info = {
#                     par: float(df.loc[i+1, par]) for par in [col for col in df.columns if any([string in col.lower() for string in ['id', 'weight', 'load', 'position', 'timestamp']])]
#                     }
#                 info['Athlet'] = df.loc[i+1, 'Athlet']
#                 xt = df.loc[j+1:].iloc[:,:2].rename(columns={col: df.loc[j,col] for col in df.iloc[:,:2]}).astype(float)# plt.plot(xt['Timestamp'], xt['Position'])
#                 df = df.drop(df.loc[i:].index)
#                 if info['V Zero Timestamp'] >= xt['Timestamp'].max() or len(xt[((xt['Position']>0.9*info['Startposition']) & (xt['Timestamp'] > info['Deep Timestamp']))]) == 0:
#                     continue
#                 # break
#                 xt['velocity'] = np.gradient(
#                     filter_signal(
#                         xt['Position'],
#                         1000,
#                         low_pass=10
#                         )
#                     )
#                 xt['acceleration'] = 1000 * np.gradient(xt['velocity'])
#                 xt_conc = get_start_by_threshold(xt, info)
#                 xt_prop = xt_conc[xt_conc['acceleration'] > 0]
#                 if all([len(x)==0 for x in [xt_conc, xt_prop]]):
#                     continue
#                 reps[rep_nr] = {**info}# {}
#                 reps[rep_nr]['rep_datetime'] = str(
#                     datetime.combine(rep_date, rep_time)
#                     )
#                 # if 'press' in info['Athlet']:
#                 #     reps[rep_nr]['exercise'] = 'press'
#                 # elif 'pull' in info['Athlet']:
#                 #     reps[rep_nr]['exercise'] = 'pull'
#                 # else:
#                 #     reps[rep_nr]['exercise'] = 'unknown'#input('enter exercise (press or pull). Tip: '+ info['Athlet']+' : ')
#                 if info['Startposition'] > 800:
#                     reps[rep_nr]['exercise'] = 'press'
#                 else:
#                     reps[rep_nr]['exercise'] = 'pull'

#                 reps[rep_nr]['xt_full'] = xt
#                 reps[rep_nr]['xt_conc'] = xt_conc
#                 reps[rep_nr]['xt_prop'] = xt_prop
    
#                 # x-t plots
#                 plt.close('all')
#                 figs[rep_nr] = {}
#                 figs[rep_nr]['xt'] = plt.figure()
#                 plt.title('rep '+str(rep_nr))
#                 plt.plot(xt['Timestamp'], xt['Position'], 'grey', linestyle=':')# reps[rep_nr]
#                 plt.plot(xt_conc['Timestamp'], xt_conc['Position'], 'blue')#, linewidth=2)#, 'grey', linestyle=':')# reps[rep_nr]
#                 plt.plot(xt_prop['Timestamp'], xt_prop['Position'], 'purple')#, linewidth=2)#, 'grey', linestyle=':')# reps[rep_nr]
#                 plt.hlines(
#                     y=[info['Startposition'], xt_prop['Position'].max()],
#                     xmin=xt_conc['Timestamp'].min()-50,#0,#
#                     xmax=xt_conc['Timestamp'].max()+50,
#                     linestyle='--',
#                     color='black'
#                     )
#                 plt.vlines(
#                     x=xt_conc['Timestamp'].min(),
#                     ymin=xt['Position'].min()-20,
#                     ymax=info['Startposition']+20,
#                     linestyle='--',
#                     color='black'
#                     )
#                 # plt.legend()
#                 plt.ylabel('position (mm)')
#                 plt.xlabel('timepoint (ms)')
#                 plt.xlim((
#                     2*xt_conc['Timestamp'].min() - plt.xlim()[1],
#                     plt.xlim()[1]
#                     ))
#                 # v-t plots
#                 figs[rep_nr]['vt'] = plt.figure()
#                 plt.title('rep '+str(rep_nr))
#                 plt.plot(xt['Timestamp'], xt['velocity'], 'grey', linestyle=':')# reps[rep_nr]
#                 plt.plot(xt_conc['Timestamp'], xt_conc['velocity'], 'orange')#, linestyle=':')# reps[rep_nr]
#                 plt.plot(xt_prop['Timestamp'], xt_prop['velocity'], 'purple')#, linewidth=2)#, 'grey', linestyle=':')# reps[rep_nr]
#                 plt.hlines(
#                     y=0,
#                     xmin=xt['Timestamp'].min(),
#                     xmax=xt['Timestamp'].max(),
#                     linewidth=1,
#                     color='black'
#                     )
#                 # plt.legend()
#                 plt.ylabel('velocity (m/s)')
#                 plt.xlabel('timepoint (ms)')
#                 plt.xlim((
#                     2*xt_conc['Timestamp'].min() - plt.xlim()[1],
#                     plt.xlim()[1]
#                     ))
#                 # a-t plots
#                 figs[rep_nr]['at'] = plt.figure()
#                 plt.title('rep '+str(rep_nr))
#                 plt.plot(xt['Timestamp'], xt['acceleration'], 'grey', linestyle=':')# reps[rep_nr]
#                 plt.plot(xt_conc['Timestamp'], xt_conc['acceleration'], 'orange')#, linestyle=':')# reps[rep_nr]
#                 plt.plot(xt_prop['Timestamp'], xt_prop['acceleration'], 'purple')#, linestyle=':')# reps[rep_nr]
#                 plt.hlines(
#                     y=0,
#                     xmin=xt['Timestamp'].min(),
#                     xmax=xt['Timestamp'].max(),
#                     linewidth=1,
#                     color='black'
#                     )
#                 # plt.legend()
#                 plt.ylabel('acceleration (m/s/s)')
#                 plt.xlabel('timepoint (ms)')
#                 plt.xlim((
#                     2*xt_conc['Timestamp'].min() - plt.xlim()[1],
#                     plt.xlim()[1]
#                     ))
#         reps, figs = get_rep_kinetics(reps, figs, new_loads={rep_nr: reps[rep_nr]['Load'] for rep_nr in reps})
#         return reps, figs

#     def get_rep_kinetics(reps, figs, new_loads):
#         for rep_nr in reps:
#             for par in ['Ft', 'Pt']:
#                 if par in figs[rep_nr]:
#                     del figs[rep_nr][par]
#             for xt in ['xt_full', 'xt_conc', 'xt_prop']:
#                 for par in ['net_Force', 'Force', 'Power']:
#                     if par in reps[rep_nr][xt].columns:
#                         reps[rep_nr][xt].drop(columns=[par], inplace=True)
#                 reps[rep_nr][xt]['net_Force'] = new_loads[rep_nr] * reps[rep_nr][xt]['acceleration']
#                 reps[rep_nr][xt]['Force'] = new_loads[rep_nr] * (reps[rep_nr][xt]['acceleration'] + 9.81)#info['load_mass'] * (xt['acceleration'] + 9.81)#xt['net_Force'] + info['F Mean load']
#                 reps[rep_nr][xt]['Power'] = reps[rep_nr][xt]['Force'] * reps[rep_nr][xt]['velocity']
#             # F-t plots
#             figs[rep_nr]['Ft'] = plt.figure()
#             plt.title('rep '+str(rep_nr))
#             plt.plot(reps[rep_nr]['xt_full']['Timestamp'], reps[rep_nr]['xt_full']['Force'], 'grey', linestyle=':')# reps[rep_nr]
#             plt.plot(reps[rep_nr]['xt_conc']['Timestamp'], reps[rep_nr]['xt_conc']['Force'], 'red')#, linestyle=':')# reps[rep_nr]
#             plt.plot(reps[rep_nr]['xt_prop']['Timestamp'], reps[rep_nr]['xt_prop']['Force'], 'purple')#, linestyle=':')# reps[rep_nr]
#             plt.ylabel('Force (N)')
#             plt.xlabel('timepoint (ms)')
#             plt.xlim((
#                 2*reps[rep_nr]['xt_conc']['Timestamp'].min() - plt.xlim()[1],
#                 plt.xlim()[1]
#                 ))
#             # P-t plots
#             figs[rep_nr]['Pt'] = plt.figure()
#             plt.title('rep '+str(rep_nr))
#             plt.plot(reps[rep_nr]['xt_full']['Timestamp'], reps[rep_nr]['xt_full']['Power'], 'grey', linestyle=':')# reps[rep_nr]
#             plt.plot(reps[rep_nr]['xt_conc']['Timestamp'], reps[rep_nr]['xt_conc']['Power'], 'green')#, linestyle=':')# reps[rep_nr]
#             plt.plot(reps[rep_nr]['xt_prop']['Timestamp'], reps[rep_nr]['xt_prop']['Power'], 'purple')#, linestyle=':')# reps[rep_nr]
#             plt.ylabel('Power (W)')
#             plt.xlabel('timepoint (ms)')
#             plt.xlim((
#                 2*reps[rep_nr]['xt_conc']['Timestamp'].min() - plt.xlim()[1],
#                 plt.xlim()[1]
#                 ))
#         return reps, figs

#     reps, figs = get_rep_kinematics(upload_file)

#     @st.cache_data
#     def post_plots(reps, pars=['xt', 'vt'], expand=[False, True]):
#         # loads = sorted(list(set([reps[rep_nr]['Load'] for rep_nr in reps])))
#         loads = {}
#         if len(expand) != len(pars):
#             expand = [False for par in pars]
#         for i,par in enumerate(pars):
#             with st.expander(par+' plots', expanded=expand[i]):
#                 for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps])):# break
#                     with st.container():
#                         st.divider()
#                         st.subheader(exercise)
#                         loads[exercise] = sorted(list(set([reps[rep_nr]['Load'] for rep_nr in reps if reps[rep_nr]['exercise']==exercise])))
#                         n_cols = len(loads[exercise])
#                         cols = st.columns(n_cols)# st.columns(len(reps))
#                         for c,load in enumerate(loads[exercise]):# break
#                             with cols[c]:
#                                 for rep_nr in [rep_nr for rep_nr in reps if reps[rep_nr]['Load']==load and reps[rep_nr]['exercise']==exercise]:
#                                     st.pyplot(figs[rep_nr][par])
                        
#     # # post_plots(reps, pars=['xt', 'vt', 'Ft'], expand=[False, True, False])
#     # post_plots(reps, pars=['xt', 'vt', 'at'], expand=[False, True, False])

#     with st.sidebar.form(key='rep_loads'):#, clear_on_submit=True):
#         new_loads = {}
#         # for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in list(reps.keys())[1:]])):
#         for exercise in [reps[1]['exercise']] + list(set([reps[rep_nr]['exercise'] for rep_nr in reps if reps[rep_nr]['exercise'] != reps[1]['exercise']])):
#             with st.container():
#                 st.subheader('adjust rep loads')
#                 st.divider()
#                 st.subheader(exercise)
#                 new_loads[exercise] = {
#                     rep_nr: st.slider(
#                         'rep '+str(rep_nr)+' mass (kg)',
#                         min_value=5,
#                         max_value=100,
#                         value=int(reps[rep_nr]['Load']),
#                         step=1,
#                         ) for rep_nr in reps if reps[rep_nr]['exercise']==exercise
#                     }
#         # '''
#         # new_loads = {
#         #         "pull":{
#         #         "9":10,
#         #         "10":10,
#         #         "11":21,
#         #         "12":21,
#         #         "13":31,
#         #         "14":31,
#         #         "15":41,
#         #         "16":41,
#         #         },
#         #         "press":{
#         #         "1":10,
#         #         "2":10,
#         #         "3":20,
#         #         "4":20,
#         #         "5":31,
#         #         "6":31,
#         #         "7":41,
#         #         "8":41,
#         #         }
#         #         }
#         # '''
#         confirm_loads = st.form_submit_button(label='confirm loads')#, on_click=get_rep_kinetics)#, on_click=get_rep_kinetics, args=(reps, figs, rep_loads))
#     if confirm_loads:
#         # st.write(new_loads)
#         # st.write(
#         #     {
#         #         **new_loads['press'],
#         #         **new_loads['pull'],
#         #         }
#         #     )
#         st.write(new_loads)
#         st.write(dict(new_loads))
#         # st.write(
#         #     {
#         #         exercise: dict(new_loads[exercise]) for exercise in new_loads
#         #         }
#         #     )
#         reps, figs = get_rep_kinetics(
#             reps, figs,
#             {
#                 # **new_loads['press'],
#                 # **new_loads['pull'],
#                 exercise: dict(new_loads[exercise]) for exercise in new_loads
#                 }
#             )
#         post_plots(reps, pars=['Ft'], expand=[True])
        
#     # if 'Force' in reps[1]['xt_conc']:
#     #     # cols = st.columns(2)
#     #     # Fv_figs = {}
#     #     # for c, exercise in enumerate(list(set([reps[rep_nr]['exercise'] for rep_nr in reps]))):
#     #     #     Fv_figs[exercise] = px.scatter(
#     #     #         data_frame=pd.DataFrame(
#     #     #             [reps[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in reps if reps[rep_nr]['exercise']==exercise]
#     #     #             ),
#     #     #         x='velocity', y='Force'
#     #     #         )
#     #     #     for rep_nr in reps if reps[rep_nr]['exercise']==exercise]:
#     #     #         Fv_fig.add_annotation(
#     #     #             x=reps[rep_nr]['xt_conc']['velocity'].mean(),
#     #     #             y=reps[rep_nr]['xt_conc']['Force'].mean(),
#     #     #             text=rep_nr
#     #     #             )
#     if 'reps_filtered' not in st.session_state:
#         st.session_state.reps_filtered = {**reps}

#     @st.cache_data
#     def filter_reps(reps, selected_reps):
#         reps_filtered = {}
#         # for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps])):
#         #     reps_filtered[exercise] = {rep_nr: reps[rep_nr] for rep_nr in reps if reps[rep_nr]['exercise']==exercise and selected_reps[exercise][rep_nr]}
#         for rep_nr in reps:
#             for exercise in selected_reps:
#                 if rep_nr in selected_reps[exercise]:
#                     if selected_reps[exercise][rep_nr]:# == True
#                         reps_filtered[rep_nr] = reps[rep_nr].copy()
#         # st.write(reps_filtered[rep_nr])
#         return reps_filtered
#     with st.form('selected reps'):
#         st.subheader('deselect sub-optimal reps')
#         selected_reps = {}
#         cols = st.columns(2)
#         for c,exercise in enumerate(list(set([reps[rep_nr]['exercise'] for rep_nr in reps]))):
#             # with st.container():
#             with cols[c]:
#                 st.divider()
#                 st.subheader(exercise)
#                 rep_nrs = sorted([rep_nr for rep_nr in reps if reps[rep_nr]['exercise']==exercise])
#                 rep_loads = [reps[rep_nr]['Load'] for rep_nr in reps if reps[rep_nr]['exercise']==exercise]
#                 loads = sorted(list(set(rep_loads)))
#                 subcols = st.columns(len(loads))
#                 selected_reps[exercise] = {
#                     rep_nrs[i]: subcols[loads.index(rep_load)].checkbox('rep_'+str(rep_nrs[i]), True, key='rep_'+str(rep_nrs[i])) for i,rep_load in enumerate(rep_loads)
#                     }
#         # st.columns(1)
#         submit_rep_selection = st.form_submit_button('confirm rep selection')
#     if submit_rep_selection:
#         st.session_state.selected_reps = selected_reps
#         st.session_state.reps_filtered = filter_reps(reps, selected_reps)
#         # st.write(reps_filtered)
#     Fv_figs_filtered = {}
#     for exercise in list(set([reps[rep_nr]['exercise'] for rep_nr in reps])): 
#         Fv_figs_filtered[exercise] = px.scatter(
#             data_frame=pd.DataFrame(
#                 # [reps[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in reps if selected_reps[rep_nr]]
#                 [st.session_state.reps_filtered[rep_nr]['xt_conc'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
#                 ),
#             x='velocity', y='Force',
#             # color='orange'
#             )
#         for rep_nr in [rep_nr for rep_nr in st.session_state.reps_filtered if reps[rep_nr]['exercise']==exercise]:
#             Fv_figs_filtered[exercise].add_annotation(
#                 x=st.session_state.reps_filtered[rep_nr]['xt_conc']['velocity'].mean(),
#                 y=st.session_state.reps_filtered[rep_nr]['xt_conc']['Force'].mean(),
#                 text=rep_nr,
#                 bgcolor='orange'
#                 )
#         Fv_figs_filtered[exercise].add_trace(
#             px.scatter(
#                 # x=[st.session_state.reps_filtered[rep_nr]['xt_prop']['velocity'].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise],
#                 # y=[st.session_state.reps_filtered[rep_nr]['xt_prop']['Force'].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise],
#                 data_frame=pd.DataFrame(
#                     [st.session_state.reps_filtered[rep_nr]['xt_prop'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
#                     ),
#                 x='velocity', y='Force',
#                 # color='purple'
#                 ).data[0]
#             )
#         for rep_nr in [rep_nr for rep_nr in st.session_state.reps_filtered if reps[rep_nr]['exercise']==exercise]:
#             Fv_figs_filtered[exercise].add_annotation(
#                 x=st.session_state.reps_filtered[rep_nr]['xt_prop']['velocity'].mean(),
#                 y=st.session_state.reps_filtered[rep_nr]['xt_prop']['Force'].mean(),
#                 text=rep_nr,
#                 bgcolor='purple'
#                 )
#         # st.write(
#         #     pd.DataFrame(
#         #         [st.session_state.reps_filtered[rep_nr]['xt_prop'][['velocity', 'Force']].mean() for rep_nr in st.session_state.reps_filtered if st.session_state.reps_filtered[rep_nr]['exercise']==exercise]
#         #         )
#         #     )
#     cols = st.columns(2)
#     for c,exercise in enumerate(Fv_figs_filtered):
#         cols[c].write(exercise)
#         cols[c].plotly_chart(Fv_figs_filtered[exercise], use_container_width=True)


#     st.header('results')
#     # st.write(st.session_state.reps_filtered[sorted(st.session_state.reps_filtered.keys())[0]])
#     # @st.cache_data
#     # def post_results(reps_filtered):
#     #     reps_propulsive = {**reps_filtered}
#     #     for rep_nr in reps_propulsive:
#     #         reps_propulsive[rep_nr]['xt_conc'] reps_propulsive[rep_nr]['xt_conc'][
#     #             reps_propulsive[rep_nr]['xt_conc']['acceleration'] > 0
#     #             ]
#     data_frame = pd.DataFrame(
#         {
#             'exercise': [st.session_state.reps_filtered[rep_nr]['exercise'] for rep_nr in st.session_state.reps_filtered],
#             'rep_nr': [rep_nr for rep_nr in st.session_state.reps_filtered],
#             'load (kg)': [st.session_state.reps_filtered[rep_nr]['Load'] for rep_nr in st.session_state.reps_filtered],

#             'conc_dist (mm)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Position'].diff().sum() for rep_nr in st.session_state.reps_filtered],
#             'conc_time (ms)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Timestamp'].diff().sum() for rep_nr in st.session_state.reps_filtered],

#             'prop_dist (mm)': [st.session_state.reps_filtered[rep_nr]['xt_prop']['Position'].diff().sum() for rep_nr in st.session_state.reps_filtered],
#             'prop_time (ms)': [st.session_state.reps_filtered[rep_nr]['xt_prop']['Timestamp'].diff().sum() for rep_nr in st.session_state.reps_filtered],

#             'conc_vel (m/s)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['velocity'].mean() for rep_nr in st.session_state.reps_filtered],
#             'peak_vel (m/s)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['velocity'].max() for rep_nr in st.session_state.reps_filtered],
#             'conc_force (N)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Force'].mean() for rep_nr in st.session_state.reps_filtered],
#             'peak_force (N)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Force'].max() for rep_nr in st.session_state.reps_filtered],
#             'conc_power (W)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Power'].mean() for rep_nr in st.session_state.reps_filtered],
#             'peak_power (W)': [st.session_state.reps_filtered[rep_nr]['xt_conc']['Power'].max() for rep_nr in st.session_state.reps_filtered],
#             },
#         index=range(len([rep_nr for rep_nr in st.session_state.reps_filtered]))
#         )
#     st.write(data_frame)
    
#     def generate_excel(df, download_filename, link_text = 'Download Excel'):
#         # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
#         output = BytesIO()
#         with pd.ExcelWriter(output, date_format='dd.mm.yyyy') as writer:
#             df.to_excel(writer, sheet_name='explo', index=False)
#             try:
#                 for col in df.columns:
#                     col_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
#                     col_idx = df.columns.get_loc(col)
#                     writer.sheets['abs'].set_column(col_idx, col_idx, col_length)
#                     writer.sheets['rel'].set_column(col_idx, col_idx, col_length)
#             except:
#                 pass
#             writer.save()
#             processed_data = output.getvalue()
            
#         b64 = base64.b64encode(processed_data)
#         link_text = 'Download Excel'
#         if not download_filename.endswith('.xlsx'):
#             download_filename = download_filename+'.xlsx'
#         return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

#     st.markdown(
#         generate_excel(data_frame, 'results_explo_'+reps[list(reps.keys())[0]]['rep_datetime'].replace(':','.')),
#         unsafe_allow_html=True
#         )#, sign_digits=3
