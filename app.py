# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:26:38 2021

@author: Micah Gross

"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\.spyder-py3\MLD_Databank"
# streamlit run cyccess_databank_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
import base64
import subprocess
import datetime
#%%
def get_id_columns():
    id_columns = ['AthleteName', 'Group', 'Group_2', 'TestType', 'TestDate', 'TestYear', 'TestMonth', 'TestDay', 'BodyMass']
    return id_columns

def get_iso_columns(angles=[70, 100]):
    iso_columns = []
    for ang in angles:
        iso_columns.append('Fmax_'+str(ang)+'_bilateral')
        iso_columns.append('Fmax_'+str(ang)+'_left')
        iso_columns.append('Fmax_'+str(ang)+'_right')
        iso_columns.append('bilateral_deficit_'+str(ang))
        iso_columns.append('LR-imbalance_'+str(ang))
    return iso_columns

def get_loadedjump_columns(loads=[0,20,40,60,80,100]):
    loadedjump_columns = []
    for jump in ['CMJ', 'SJ']:
        for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'F1/3']:
            for ld in [str(x) for x in loads]:
                loadedjump_columns.append('_'.join([jump, par, ld]))
    return loadedjump_columns

def get_singlejump_columns():
    singlejump_columns = []
    for jump in ['CMJ', 'SJ']:
        for side in ['0', 'left','right']:
            for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'F1/3']:
                singlejump_columns.append('_'.join([jump, par, side]))
    return singlejump_columns
    
#%%
def generate_excel(db, db_rel):
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        db.to_excel(writer,sheet_name='abs',index=False)
        db_rel.to_excel(writer,sheet_name='rel',index=False)
        for col in db.columns:
            col_length = max(db[col].astype(str).map(len).max(), len(col)) + 2
            col_idx = db.columns.get_loc(col)
            writer.sheets['abs'].set_column(col_idx, col_idx, col_length)
            writer.sheets['rel'].set_column(col_idx, col_idx, col_length)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    download_filename = 'MLD_Databank.xlsx'
    link_text = 'Download databank'
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Excel databank</a>' # decode b'abc' => abc
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

#%%
st.write("""

# Web app for transforming MLD csv-export files to excel database format

""")

# with pd.ExcelWriter(os.path.join(os.getcwd(), 'test_columns.xlsx')) as writer:
#     pd.DataFrame(columns=all_columns).to_excel(writer, index=False)
#     writer.save()

# with pd.ExcelWriter(os.path.join(os.getcwd(), 'test_columns.xlsx')) as writer:
#     db.to_excel(writer, index=False)
#     writer.save()

data_export_files = st.file_uploader("upload csv export file", accept_multiple_files=True)
if data_export_files is not None and len(data_export_files)>0:
    id_columns = get_id_columns()
    iso_columns = get_iso_columns()
    loadedjump_columns = get_loadedjump_columns()
    singlejump_columns = get_singlejump_columns()
    all_columns = list(pd.Series(id_columns + iso_columns + loadedjump_columns + singlejump_columns).unique())
    bm_iso = {}
    bm_vertjump = {}
    bm_vertjump = {}
    db = pd.DataFrame()
    for f_nr,f in enumerate(data_export_files):# f = data_export_files[0]
        # with open(os.path.join(os.getcwd(),'saved_variables','.'.join(f.name.split('.')[:-1])+'_bytesIO.txt'), 'wb') as fp:
        #     fp.write(f.getbuffer())
        df = pd.read_csv(f, sep=';', encoding='cp1252')

        first_name, last_name, sex, birth_date, group, subgroup = list(df.iloc[0,:6])
        test_type, test_date, body_mass = list(df.iloc[1,6:9])
        test_date = datetime.date(year=int(test_date.split('.')[2]), month=int(test_date.split('.')[1]), day=int(test_date.split('.')[0]))
        idx = '_'.join([first_name, last_name, str(test_date)])
        db.loc[idx,'AthleteName'] = first_name+' '+last_name
        db.loc[idx,'Group'] = subgroup+' '+group
        db.loc[idx,'Group_2'] = ''
        if 'TestType' not in db.columns:
            db.loc[idx,'TestType'] = ''
        db.loc[idx,'TestDate'] = test_date
        db.loc[idx,'TestYear'] = test_date.year
        db.loc[idx,'TestMonth'] = test_date.month
        db.loc[idx,'TestDay'] = test_date.day
        db.loc[idx,'BodyMass'] = body_mass

        if test_type == 'Isometrische Maximalkraft':
            bm_iso[idx] = body_mass
            if db.loc[idx,'TestType'] =='' or type(db.loc[idx,'TestType'])!=str:
                db.loc[idx,'TestType'] = test_type
            else:
                db.loc[idx,'TestType'] = str(db.loc[idx,'TestType']) + ', ' + test_type
            for ang in [70, 100]:
                db.loc[idx,'Fmax_'+str(ang)+'_bilateral'] = df[((df['Winkel_iso [°]']==ang) & (df['Ausführung']!='einbeinig links') & (df['Ausführung']!='einbeinig rechts'))]['Fmax_iso [N]'].mean()
                db.loc[idx,'Fmax_'+str(ang)+'_left'] = df[((df['Winkel_iso [°]']==ang) & (df['Ausführung']=='einbeinig links'))]['Fmax_iso [N]'].mean()
                db.loc[idx,'Fmax_'+str(ang)+'_right'] = df[((df['Winkel_iso [°]']==ang) & (df['Ausführung']=='einbeinig rechts'))]['Fmax_iso [N]'].mean()
                db.loc[idx,'bilateral_deficit_'+str(ang)] = 1 - (db.loc[idx,'Fmax_'+str(ang)+'_bilateral'] / (db.loc[idx,'Fmax_'+str(ang)+'_left'] + db.loc[idx,'Fmax_'+str(ang)+'_right']))# calculate bilateral deficit
                db.loc[idx,'LR-imbalance_'+str(ang)] = 100*(1-np.min([db.loc[idx,'Fmax_'+str(ang)+'_left'], db.loc[idx,'Fmax_'+str(ang)+'_right']])/np.max([db.loc[idx,'Fmax_'+str(ang)+'_left'], db.loc[idx,'Fmax_'+str(ang)+'_right']]))

        if test_type == 'LoadedJump':
            bm_vertjump[idx] = body_mass
            if db.loc[idx,'TestType'] =='' or type(db.loc[idx,'TestType'])!=str:
                db.loc[idx,'TestType'] = test_type
            else:
                db.loc[idx,'TestType'] = str(db.loc[idx,'TestType']) + ', ' + test_type
            execution_types = ['elastodyn', 'statodyn']
            for j,jump in enumerate(['CMJ', 'SJ']):# jump='CMJ'
                for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'F1/3']:# par='Pmax'
                    for ld in [str(x) for x in [0,20,40,60,80,100]]:# ld=str(0)
                        db.loc[idx,'_'.join([jump,par,ld])] = df[(
                            (df['Ausführung']==execution_types[j]) & (90+float(ld) < df['%KG [%]']) & (df['%KG [%]'] < float(ld)+110)
                            )][
                                [col for col in df.columns if col.startswith(par)][0]
                                ].mean()
         
        if test_type == 'Einzelsprung':
            bm_vertjump[idx] = body_mass
            if db.loc[idx,'TestType'] =='' or type(db.loc[idx,'TestType'])!=str:
                db.loc[idx,'TestType'] = test_type
            else:
                db.loc[idx,'TestType'] = str(db.loc[idx,'TestType']) + ', ' + test_type
            execution_types = ['elastodyn', 'einbeinig links', 'einbeinig rechts']
            for jump in ['CMJ']:# jump='CMJ'
                for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'F1/3']:# par='Pmax'
                    for s,side in enumerate(['0', 'left','right']):
                        db.loc[idx,'_'.join([jump,par,side])] = df[
                            df['Ausführung']==execution_types[s]
                            ][
                                [col for col in df.columns if col.startswith(par)][0]
                                ].mean()
            for jump in ['SJ']:# jump='CMJ'
                for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'F1/3']:# par='Pmax'
                    for ld in ['0']:
                        db.loc[idx,'_'.join([jump,par,ld])] = df[
                            df['Ausführung']=='statodyn'
                            ][
                                [col for col in df.columns if col.startswith(par)][0]
                                ].mean()
            # 'Pmax_CMJ_left', 'Pavg_CMJ_left', 'h_CMJ_left', 'load_CMJ_left', 'spos_CMJ_left', 'tpos_CMJ_left', 'Fmax_CMJ_left', 'vmax_CMJ_left', 'Fv0_CMJ_left', 'F1/3_CMJ_left', 'Pmax_CMJ_right', 'Pavg_CMJ_right', 'h_CMJ_right', 'load_CMJ_right', 'spos_CMJ_right', 'tpos_CMJ_right', 'Fmax_CMJ_right', 'vmax_CMJ_right', 'Fv0_CMJ_right', 'F1/3_CMJ_right'
    
    for col in all_columns:
        if col not in db.columns:
            db.insert(len(db.columns), col, np.nan)
            
    db = db[all_columns]
    db_rel = db.copy()
    for par in ['Fmax_70', 'Fmax_100']:
        for col in db_rel.columns:
            if col.startswith(par):
                for i in db_rel.index:
                    try:# only works if the iso test was performed
                        db_rel.loc[i,col] = db.loc[i,col] / bm_iso[i]
                    except:
                        pass
    for par in ['_Pmax_', '_Ppos_', '_Fmax_', '_Fv0_', '_F1/3_']:
        for col in db_rel.columns:
            if par in col:
                for i in db_rel.index:
                    try:# works if either vertical jump test was performed
                        db_rel.loc[i,col] = db.loc[i,col] / bm_vertjump[i]
                    except:
                        pass
        
    st.markdown(
        generate_excel(db, db_rel),
        unsafe_allow_html=True
        )#, sign_digits=3
        
#%%
def retrieve_variables():
    for (_, _, file_list) in os.walk(os.path.join(os.getcwd(),'saved_variables')):# get main path and list of directories
        break
    data_export_files = []
    data_export_file_names = [x for x in file_list if '_bytesIO.txt' in x]
    for f in data_export_file_names:
        with open(os.path.join(os.getcwd(),'saved_variables',f), 'rb') as fh:
            file = BytesIO(fh.read())
            data_export_files.append(file)
    # del f, fh, file, file_list, data_export_file_names
    return data_export_files, data_export_file_names

# data_export_files, data_export_file_names = retrieve_variables()

