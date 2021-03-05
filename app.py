# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:49:51 2021

@author: Micah Gross

"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\.spyder-py3\MLD_Databank"
# streamlit run mld_databank_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
# import xlsxwriter
# import openpyxl
# import xlwings
from io import BytesIO
import base64
import subprocess

# from MLD_Databank_v8_addedSingleJump import create_db_raw
##%%
def create_db_raw(data_export_file):# data_export_file = data_export_files[0]
    # import first column (test identifiers) of csv-file into DataFrame
    data_export_file.seek(0)
    df_col1 = pd.read_csv(data_export_file, delimiter=';', header=None, names=['AthleteName'], usecols=[0])# first column
    NextTestRows = []# list of row numbers in csv where a test begins
    for i in range(len(df_col1)):
        try:
            int(df_col1.iloc[i][0][0])# if the cell contains a number (a date)...
            NextTestRows.append(i-3)# ...add the row number-3 to the list
        except:
            pass# if not, pass
    # split information from column 1 into separate columns
    df_col1['Group'] = df_col1['AthleteName']
    for i in range(len(df_col1)-1):
        df_col1['Group'][i] = df_col1['Group'][i+1]
    df_col1['TestForm'] = df_col1['Group']
    for i in range(len(df_col1)-1):
        df_col1['TestForm'][i] = df_col1['TestForm'][i+1]
    df_col1['TestDate'] = df_col1['TestForm']
    for i in range(len(df_col1)-1):
        df_col1['TestDate'][i] = df_col1['TestDate'][i+1]
    # fill out intermediate rows with nan
    for i in range(len(df_col1)):
        if i not in NextTestRows:
            df_col1.iloc[i] = np.full((1, 4), np.nan).tolist()[0]
    # convert TestDate into numericals for year, month, and day
    df_col1['TestDateAndTime'] = df_col1['TestDate']
    df_col1['TestYear'] = df_col1['TestDate']
    df_col1['TestMonth'] = df_col1['TestDate']
    df_col1['TestDay'] = df_col1['TestDate']
    df_col1['TestTime'] = df_col1['TestDate']
    for i in NextTestRows:
        df_col1['TestDate'][i] = df_col1['TestTime'][i][:df_col1['TestTime'][i].index(' ')]# date only, without time
        df_col1['TestTime'][i] = df_col1['TestTime'][i][df_col1['TestTime'][i].index(' ')+1:]# time only, without date
        df_col1['TestYear'][i] = int(df_col1['TestDate'][i][df_col1['TestDate'][i].rindex('.')+1:])# year as a number
        df_col1['TestMonth'][i] = int(df_col1['TestDate'][i][df_col1['TestDate'][i].index('.')+1:df_col1['TestDate'][i].index('.')+3])# month as a number
        df_col1['TestDay'][i] = int(df_col1['TestDate'][i][:df_col1['TestDate'][i].index('.')])# day as a number
        # reconstruct TestDate in reversed-order format (yyyy.mm.dd) for sorting purposes
        df_col1['TestDate'][i] = '.'.join(
            [
                str(df_col1['TestYear'][i]),
                str(df_col1['TestMonth'][i]) if len(str(df_col1['TestMonth'][i]))==2 else str(0)+str(df_col1['TestMonth'][i]),
                str(df_col1['TestDay'][i]) if len(str(df_col1['TestDay'][i]))==2 else str(0)+str(df_col1['TestDay'][i])
                ]
            )# date in reversed-order format (yyyy.mm.dd)
    # import remaining columns (2 and beyond; actual data) of csv-file into DataFrame
    data_export_file.seek(0)
    df_cols = pd.read_csv(data_export_file, delimiter=';', header=4, usecols=range(1,202))
    # combine DataFrames into on
    df = pd.concat([df_col1, df_cols], axis=1)
    # del df_col1, df_cols # delete partial DataFrames
    df.rename(columns ={'Unnamed: 1': 'Measurement'}, inplace=True)
    # delete all rows containing headers, except first row
    for i in NextTestRows[1:]:
        df = df.drop([i-1])
    # delete all rows containing nothing but nan
    for i in df.index:
        if df.loc[i].isnull().values.all():
            df = df.drop([i])
    # delete all columns containing nothing but nan
    for i in df.columns:
        if df[i].isnull().values.all():
            df = df.drop([i], axis=1)
    # reset index
    df = df.reset_index(drop=True)
    # fill out all rows with corresponding test identifiers
    for i in df.index:
        if df.iloc[i,:9].isnull().values.all():
            df.iloc[i,:9] = df.iloc[i-1,:9]
    # convert actual data to numeric
    df.iloc[:,10:] = df.iloc[:,10:].apply(pd.to_numeric)
    #replace symbols in headers and measurement names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('°', 'deg')
    df.columns = df.columns.str.replace('%', 'pct')
    df.Measurement = df.Measurement.str.replace('°', 'deg')
    df.Measurement = df.Measurement.str.replace('%', 'pct')
    for col in df.columns[0:2]:
        for letter in list(zip(
                ['ä', 'ö', 'ü', 'é', 'è'],
                ['ae', 'oe', 'ue', 'e', 'e']
                )):
            df[col] = df[col].str.replace(letter[0], letter[1])
    # sort DataFrame
    df.sort_values(by=['Group', 'AthleteName', 'TestForm', 'TestYear', 'TestMonth', 'TestDay', 'Measurement'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # del NextTestRows, i
    return df

##%%
def subset(df, *args):
    '''function for extracting a subset
    sample input:
        df=df
        args=['AthleteName','Blanc Renaud']
        args=['AthleteName','Blanc Renaud','TestForm','Isometrische Maximalkraft','Measurement','IsoTest 100deg']
        args=['AthleteName','Graf David','TestForm','Isometrische Maximalkraft','Measurement','IsoTest 100deg']
    example execution:
        ss=subset(df,'AthleteName','Blanc Renaud','TestForm','Isometrische Maximalkraft','Measurement','IsoTest 100deg')
        ss=subset(df,'AthleteName','Graf David','TestForm','Isometrische Maximalkraft','Measurement','IsoTest 100deg')
        ss=df.query(AthleteName=='Cologna Dario'and TestForm=='Isometrische Maximalkraft' and Measurement=='IsoTest 100deg')
    '''    
    pairs = []
    for column, value in list(zip(args[0::2], args[1::2])):
        if args[0::2].index(column)==0:
            pairs.append('(df['+repr(str(column))+']=='+repr(str(value))+')')
        else:
            pairs.append('& (df['+repr(str(column))+']=='+repr(str(value))+')')
    subset = df.loc[eval(' '.join(pairs))]
    subset = subset.sort_values(by=['TestYear','TestMonth','TestDay'])
    return subset

def round_df(df):
    for c in df.columns:# c=df.columns[5] # c=df.columns[15] # c=df.columns[18]
        if type(df[c].iloc[0])==int:
            df[c] = df[c].astype(int)
        if 'float' in str(type(df[c].iloc[0])):# True for float, numpy.float64, etc.
            df[c] = df[c].astype(float)
            if df[c].iloc[0]>=100:
                df[c] = df[c].round(0)
            elif df[c].iloc[0]>=10:
                df[c] = df[c].round(1)
            elif df[c].iloc[0]>=1:
                df[c] = df[c].round(2)
            else:
                df[c] = df[c].round(3)
    return df# df_rounded=round_df(df)
    
##%%
def create_db_summary(db_raw, jump_parameters=['Pmax_','Pavg_','h_','load_','spos_','tpos_','Fmax_','vmax_','Fv0_','F1/3_']):
    headers = db_raw.columns# get all headers
    headers = headers[[0,1,3,5,6,7]].tolist()# reduce to selected headers
    headers.append('body_mass')
    for a in [70,100]:# angles for isometric force test
        for s in ['_bilateral', '_left', '_right']:# positions for isometric force test
            headers.append('Fmax_' + str(a) + s)
        for p in ['bilateral_deficit_', 'LR-imbalance_']:
            headers.append(p + str(a))
    for j in ['CMJ_', 'SJ_']:# loaded jump forms
        for p in jump_parameters:
            for pct in [0,20,40,60,80,100]:
                headers.append(p + j + str(pct))
    for j in ['CMJ_left', 'CMJ_right']:# other jump forms
        for p in jump_parameters:
            headers.append(p + j)
    db_summary = pd.DataFrame(columns=headers)
    # list of selected column headers of db_summary
    a = db_summary.columns[[i for i in list(range(7,10)) + list(range(12,15)) + list(range(17, len(db_summary.columns)))]].tolist()
    # list of corresponding labels from db_raw.Measurement
    b = sum(
        [
            ['IsoTest 70deg','IsoTest 70deg left','IsoTest 70deg right'],# Fmax, 70°
            ['IsoTest 100deg','IsoTest 100deg left','IsoTest 100deg right'],# Fmax, 100°
            ['Elastojump 100pct','Elastojump 120pct','Elastojump 140pct','Elastojump 160pct','Elastojump 180pct','Elastojump 200pct']*len(jump_parameters),# CMJ
            ['Statojump 100pct','Statojump 120pct','Statojump 140pct','Statojump 160pct','Statojump 180pct','Statojump 200pct']*len(jump_parameters),# SJ
            ['Single Jump left']*len(jump_parameters),
            ['Single Jump right']*len(jump_parameters),
            ],
        []
        )
    # list of corresponding labels from db_raw.columns
    c = sum(
        [
            ['Fmax_iso_[N]']*3,# isometric 70°
            ['Fmax_iso_[N]']*3,# isometric 100°
            sum(
                [
                    ['Pmax_abs_[Watt]']*6,# Pmax CMJ, *6 for six loading conditions
                    ['Ppos_abs_[Watt]']*6,# Pavg CMJ
                    ['s_max_[cm]']*6,# height (h) CMJ
                    ['load_[kg]']*6,# load CMJ
                    ['s_pos_[cm]']*6,# spos CMJ
                    ['tpos_[s]']*6,# tpos CMJ
                    ['Fmax_[N]']*6,# Fmax CMJ
                    ['Vmax_[m/s]']*6,# vmax CMJ
                    ['Fv0_[N]']*6,# Fv0 CMJ
                    ['F1/3_abs_[N]']*6,# F1/3
                ],
                [])*2,# starting point for 'sum' ([]), *2 for CMJ, SJ
            sum(
                [
                    ['Pmax_abs_[Watt]'],# Pmax CMJ
                    ['Ppos_abs_[Watt]'],# Pavg CMJ
                    ['s_max_[cm]'],# height (h) CMJ
                    ['load_[kg]'],# load CMJ
                    ['s_pos_[cm]'],# spos CMJ
                    ['tpos_[s]'],# tpos CMJ
                    ['Fmax_[N]'],# Fmax CMJ
                    ['Vmax_[m/s]'],# vmax CMJ
                    ['Fv0_[N]'],# Fv0 CMJ
                    ['F1/3_abs_[N]'],# F1/3
                ],
                [])*2,
           ],# starting point for 'sum' ([]), *2 for single jump left, and single jump right
        [])# starting point for 'sum' ([])
    excluded_tests = []
    for i, testID in enumerate(list(set(zip(db_raw.AthleteName, db_raw.TestDate)))):# unique combinations of AthleteName and TestDate # [i,testID]=[0,list(set(zip(db_raw.AthleteName,db_raw.TestDate)))[0]]
        try:
            db_summary.loc[i] = np.full((1, len(db_summary.columns)), np.nan).tolist()[0]# first make a new row full of NaN
            db_summary.loc[i, 'AthleteName'] = testID[0]
            db_summary.loc[i, 'TestDate'] = testID[1]
            idx = list(zip(db_raw.AthleteName, db_raw.TestDate)).index(testID)# row where the test begins (row of first measurement) in db_raw
            for p in db_raw.columns[[1,5,6,7]]:# parameters that are the same for all measurements in that test (can be taken from the first row)
                db_summary.loc[i, p] = db_raw.loc[idx, p]
            try:# calculate body mass
                d = db_raw.loc[list(zip(db_raw.AthleteName, db_raw.TestDate, db_raw.Measurement)).index(list(zip([testID[0]], [testID[1]], ['Statojump 100pct']))[0]), 'Pmax_abs_[Watt]']
                e = db_raw.loc[list(zip(db_raw.AthleteName, db_raw.TestDate, db_raw.Measurement)).index(list(zip([testID[0]], [testID[1]], ['Statojump 100pct']))[0]), 'Pmax_rel_[Watt/kg]']
            except:
                try:
                    d = db_raw.loc[list(zip(db_raw.AthleteName, db_raw.TestDate, db_raw.Measurement)).index(list(zip([testID[0]], [testID[1]], ['Elastojump 100pct']))[0]), 'Pmax_abs_[Watt]']
                    e = db_raw.loc[list(zip(db_raw.AthleteName, db_raw.TestDate, db_raw.Measurement)).index(list(zip([testID[0]], [testID[1]], ['Elastojump 100pct']))[0]), 'Pmax_rel_[Watt/kg]']
                except:
                    d = db_raw.loc[list(zip(db_raw.AthleteName, db_raw.TestDate, db_raw.Measurement)).index(list(zip([testID[0]], [testID[1]], ['IsoTest 100deg']))[0]), 'Fmax_iso_[N]']
                    e = db_raw.loc[list(zip(db_raw.AthleteName, db_raw.TestDate, db_raw.Measurement)).index(list(zip([testID[0]], [testID[1]], ['IsoTest 100deg']))[0]), 'Fmax_iso_rel_[N/kg]']
            db_summary.loc[i, 'body_mass']=d/e# enter body mass
            for t in list(zip(a,b,c)):# t=list(zip(a,b,c))[3] # t=list(zip(a,b,c))[6]
                try:
                    # db_summary.loc[i,t[0]]=db_raw.loc[list(zip(db_raw.AthleteName,db_raw.TestDate,db_raw.Measurement)).index(list(zip([testID[0]],[testID[1]],[t[1]]))[0]),t[2]]# fill out the remaining main parameters
                    db_summary.loc[i,t[0]] = db_raw[((db_raw['AthleteName']==testID[0]) & (db_raw['TestDate']==testID[1]) & (db_raw['Measurement']==t[1]))][t[2]].mean()# take mean if multiple values for same measurement
                except:
                    pass
        except:
           excluded_tests.append(testID)

    db_summary.bilateral_deficit_100 = 1 - (db_summary.Fmax_100_bilateral / (db_summary.Fmax_100_left + db_summary.Fmax_100_right))# calculate bilateral deficit
    for ld in [str(70),str(100)]:# calculate imbalances # ld=str(100)
        for i in range(len(db_summary)):
            if db_summary.loc[i,'Fmax_'+ld+'_left']*db_summary.loc[i,'Fmax_'+ld+'_right']>0:
                db_summary.loc[i,'LR-imbalance_'+ld] = 100*(1-np.min([db_summary.loc[i,'Fmax_'+ld+'_left'],db_summary.loc[i,'Fmax_'+ld+'_right']])/np.max([db_summary.loc[i,'Fmax_'+ld+'_left'],db_summary.loc[i,'Fmax_'+ld+'_right']]))    
            else:
                pass
    db_summary = db_summary.sort_values(by=['Group', 'AthleteName', 'TestDate']).reset_index(drop=True)
    db_summary.insert(2, 'Group_3', '')
    db_summary.insert(2, 'Group_2', '')
    db_summary_rel = db_summary.copy()
    for col in db_summary.columns:# col=list(db_summary.columns)[7]
        if ((col[0]=='F') or (col[0]=='P')):
            db_summary_rel[col] = db_summary_rel[col] / db_summary_rel['body_mass']
    
    db_summary = round_df(db_summary)
    db_summary_rel = round_df(db_summary_rel)
    return db_summary, db_summary_rel

def generate_excel(db_summary, db_summary_rel, n):
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        db_summary.to_excel(writer,sheet_name='abs',index=False)
        db_summary_rel.to_excel(writer,sheet_name='rel',index=False)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    download_filename = 'MLD_Databank_'+str(n)+'.xlsx'
    link_text = 'Download databank ' + str(n)
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Excel databank</a>' # decode b'abc' => abc
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

#%%
st.write("""

# Web app for transforming MLD csv-export files to excel database format

""")
data_export_files = st.file_uploader("upload csv export file", accept_multiple_files=True)
if data_export_files is not None:
    for n,f in enumerate(data_export_files, start=1):# f = data_export_files[0]
        st.write('.'.join(f.name.split('.')[:-1]))
        # st.write('.'.join(f.name.split('.')[:-1])+'_bytesIO.txt')
        # with open(os.path.join(os.getcwd(),'saved_variables','.'.join(f.name.split('.')[:-1])+'_bytesIO.txt'), 'wb') as fp:
        #     fp.write(f.getbuffer())
        # f.seek(0)
        db_raw = create_db_raw(f)
        db_summary, db_summary_rel = create_db_summary(db_raw)# jump_parameters=['Pmax_','Pavg_','h_','load_','spos_','tpos_','Fmax_','vmax_','Fv0_','F1/3_']# list of parameters needed for jumps
        if db_summary is not None:
                # writing_excel_container = st.empty()
                # writing_excel_container.text('writing to excel: ' + '.'.join(f.name.split('.')[:-1]))
            st.markdown(
                generate_excel(db_summary, db_summary_rel, n),
                unsafe_allow_html=True
                )#, sign_digits=3


        


