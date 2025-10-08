# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:26:38 2021

@author: Micah Gross

"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\user\OneDrive\python_scripts\Streamlit\Cyccess"
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
import base64
# import subprocess
import datetime
##%%
def get_id_columns():
    id_columns = ['AthleteName', 'BirthDate', 'Group', 'Group_2', 'TestType', 'TestDate', 'TestYear', 'TestMonth', 'TestDay', 'BodyMass']
    return id_columns

def get_iso_columns(angles=[70, 100]):
    iso_columns = []
    for ang in angles:
        for par in ['bilateral', 'left', 'right', 'bilateral_deficit', 'LR-imbalance']:
            iso_columns.append('_'.join(['Fmax', str(ang), par]))
    return iso_columns

def get_loadedjump_columns(loads=[0, 20, 40, 60, 80, 100]):
    loadedjump_columns = []
    for jump in ['CMJ', 'SJ']:
        for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'P1/3', 'Fpos', 't_Fmax', 'tacc', 'tneg']:
            for ld in [str(x) for x in loads]:
                loadedjump_columns.append('_'.join([jump, par, ld]))
    return loadedjump_columns

def get_singlejump_columns():
    singlejump_columns = []
    for jump in ['CMJ', 'SJ']:
        for side in ['0', 'left','right']:
            for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'P1/3', 'Fpos', 't_Fmax', 'tacc', 'tneg']:
                singlejump_columns.append('_'.join([jump, par, side]))
    for par1 in ['effect_of_prestretch', 'bilateral_deficit', 'LR-imbalance']:
        for par2 in ['Pmax', 's_max']:
            singlejump_columns.append('_'.join(['CMJ', par2, '0', par1]))
    return singlejump_columns
    
def get_dropjump_columns(drop_heights=[20, 40, 60]):
    dropjump_columns = []
    for jump in ['DJ']:# jump = 'DJ'
        for par in ['s_max', 'tacc', 'Reak1', 'Reak2', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'Fpos', 't_Fmax', 'tacc', 'tneg']:# par = 'Reak2'
            for dh in [str(x) for x in drop_heights]:# dh = str(20)
                dropjump_columns.append('_'.join([jump, par, dh]))
    return dropjump_columns
    
##%%
def generate_excel(**kwargs):
    sheet_name_map = {'db': 'abs', 'db_rel': 'rel', 'db_alt': 'Sheet1'}
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    with pd.ExcelWriter(output, date_format='dd.mm.yyyy') as writer:
        for df_name in kwargs:
            s_name = sheet_name_map[df_name]
            df = kwargs[df_name]
            df.to_excel(writer, sheet_name=s_name, index=False)
            # db_rel.to_excel(writer,sheet_name='rel',index=False)
            try:
                for col in df.columns:
                    col_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    col_idx = df.columns.get_loc(col)
                    writer.sheets[s_name].set_column(col_idx, col_idx, col_length)
                    # writer.sheets['rel'].set_column(col_idx, col_idx, col_length)
            except:
                pass
        # writer.save()
        writer.close()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    download_filename = 'Cyccess_Databank.xlsx'
    link_text = 'Download databank'
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Excel databank</a>' # decode b'abc' => abc
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

# def generate_excel(db, db_rel):
#     # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
#     writing_excel_container = st.empty()
#     writing_excel_container.text('writing to excel')
#     output = BytesIO()
#     with pd.ExcelWriter(output, date_format='dd.mm.yyyy') as writer:
#         db.to_excel(writer,sheet_name='abs',index=False)
#         db_rel.to_excel(writer,sheet_name='rel',index=False)
#         try:
#             for col in db.columns:
#                 col_length = max(db[col].astype(str).map(len).max(), len(col)) + 2
#                 col_idx = db.columns.get_loc(col)
#                 writer.sheets['abs'].set_column(col_idx, col_idx, col_length)
#                 writer.sheets['rel'].set_column(col_idx, col_idx, col_length)
#         except:
#             pass
#         # writer.save()
#         writer.close()
#         processed_data = output.getvalue()
        
#     b64 = base64.b64encode(processed_data)
#     writing_excel_container.empty()
#     # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
#     download_filename = 'Cyccess_Databank.xlsx'
#     link_text = 'Download databank'
#     # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Excel databank</a>' # decode b'abc' => abc
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

# def generate_excel_alt(db_alt):
#     # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
#     writing_excel_container = st.empty()
#     writing_excel_container.text('writing to excel')
#     output = BytesIO()
#     with pd.ExcelWriter(output, date_format='dd.mm.yyyy') as writer:
#         db_alt.to_excel(writer, sheet_name='Sheet1' ,index=False)
#         try:
#             for col in db_alt.columns:
#                 col_length = max(db_alt[col].astype(str).map(len).max(), len(col)) + 2
#                 col_idx = db_alt.columns.get_loc(col)
#                 writer.sheets['Sheet1'].set_column(col_idx, col_idx, col_length)
#         except:
#             pass
#         # writer.save()
#         writer.close()
#         processed_data = output.getvalue()
        
#     b64 = base64.b64encode(processed_data)
#     writing_excel_container.empty()
#     # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
#     download_filename = 'Cyccess_Databank.xlsx'
#     link_text = 'Download databank'
#     # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Excel databank</a>' # decode b'abc' => abc
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc

#%%
# def retrieve_saved_variables():
#     with open(os.path.join(os.getcwd(),'saved_variables','Options.json'), 'r') as fp:
#         Options = json.load(fp)
#     for (_, _, file_list) in os.walk(os.path.join(os.getcwd(),'saved_variables')):# get main path and list of directories
#         break
#     data_export_files = []
#     data_export_file_names = [x for x in file_list if '_bytesIO.txt' in x]
#     for f in data_export_file_names:
#         with open(os.path.join(os.getcwd(),'saved_variables',f), 'rb') as fh:
#             file = BytesIO(fh.read())
#             data_export_files.append(file)
#     del fp, f, fh, file, file_list#, data_export_file_names
#     return Options, data_export_files, data_export_file_names

# # Options, data_export_files, data_export_file_names = retrieve_saved_variables()
# # Options['save_variables'] = False

def process_uploaded_files(data_export_files, Options, live=True):# data_export_files, Options, live = ["C:\\Users\\user\\OneDrive\\data_transfer\\cycexport_16102024_23122024.csv"], {'valid_only': True}, False
    id_columns = get_id_columns()
    iso_columns = get_iso_columns()
    loadedjump_columns = get_loadedjump_columns()
    singlejump_columns = get_singlejump_columns()
    dropjump_columns = get_dropjump_columns()
    all_columns = list(pd.Series(id_columns + iso_columns + loadedjump_columns + singlejump_columns + dropjump_columns).unique())
    bm_iso = {}
    bm_vertjump = {}
    bm_dropjump = {}
    db = pd.DataFrame()
    for f_nr,f in enumerate(data_export_files):# f_nr,f = 0,data_export_files[0]
        if live:
            with current_file_container:
                st.write(f.name)
        # if Options['save_variables']:
        #     with open(os.path.join(os.getcwd(),'saved_variables','.'.join(f.name.split('.')[:-1])+'_bytesIO.txt'), 'wb') as fp:
        #         fp.write(f.getbuffer())
        f.seek(0)
        df_file = pd.read_csv(f, sep=';', encoding='cp1252')
        ons = list(df_file['Vorname'].dropna().index) + [len(df_file)]
        df_multi = {
            n: df_file.loc[ons[n]:ons[n+1]-1,:] for n,i in enumerate(ons[:-1])
            }
        for key in df_multi:
            df = df_multi[key]
            if Options['valid_only']:
                df = df[df['Gültig']!='Nein'].reset_index(drop=True)
            
            first_name, last_name, sex, birth_date = list(df.iloc[0,:4])# [str(x) for x in list(df.iloc[0,:6])]
            group, subgroup = [str(x) for x in list(df.iloc[0,4:6])]
            test_type, test_date, body_mass = list(df.iloc[1,6:9])
            test_date = datetime.date(year=int(test_date.split('.')[2]), month=int(test_date.split('.')[1]), day=int(test_date.split('.')[0]))
            if type(birth_date)==str:# which means the birth date is not missing
                birth_date = datetime.date(year=int(birth_date.split('.')[2]), month=int(birth_date.split('.')[1]), day=int(birth_date.split('.')[0]))
            idx = '_'.join([first_name, last_name, str(test_date)])
            db.loc[idx,'AthleteName'] = first_name+' '+last_name
            db.loc[idx,'BirthDate'] = birth_date
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
                    db.loc[idx,'TestType'] = ', '.join(sorted([str(db.loc[idx,'TestType']), test_type]))# ', '.join(sorted[str(db.loc[idx,'TestType']), test_type])# str(db.loc[idx,'TestType']) + ', ' + test_type
                for ang in [70, 100]:# ang = 100
                    db.loc[idx,'_'.join(['Fmax', str(ang), 'bilateral'])] = df[((df['Winkel_iso [°]']==ang) & (df['Ausführung']!='einbeinig links') & (df['Ausführung']!='einbeinig rechts'))]['Fmax_iso [N]'].max()
                    db.loc[idx,'_'.join(['Fmax', str(ang), 'left'])] = df[((df['Winkel_iso [°]']==ang) & (df['Ausführung']=='einbeinig links'))]['Fmax_iso [N]'].max()
                    db.loc[idx,'_'.join(['Fmax', str(ang), 'right'])] = df[((df['Winkel_iso [°]']==ang) & (df['Ausführung']=='einbeinig rechts'))]['Fmax_iso [N]'].max()
                    db.loc[idx,'_'.join(['Fmax', str(ang), 'bilateral_deficit'])] = 100*(1 - (db.loc[idx,'Fmax_'+str(ang)+'_bilateral'] / (db.loc[idx,'Fmax_'+str(ang)+'_left'] + db.loc[idx,'Fmax_'+str(ang)+'_right'])))# calculate bilateral deficit
                    # db.loc[idx,'_'.join(['Fmax', str(ang), 'bilateral_deficit'])] = 100*(((db.loc[idx,'Fmax_'+str(ang)+'_left'] + db.loc[idx,'Fmax_'+str(ang)+'_right']) / db.loc[idx,'Fmax_'+str(ang)+'_bilateral']) - 1)# calculate bilateral deficit
                    db.loc[idx,'_'.join(['Fmax', str(ang), 'LR-imbalance'])] = 100*(1-np.min([db.loc[idx,'Fmax_'+str(ang)+'_left'], db.loc[idx,'Fmax_'+str(ang)+'_right']])/np.max([db.loc[idx,'Fmax_'+str(ang)+'_left'], db.loc[idx,'Fmax_'+str(ang)+'_right']]))
    
            elif test_type == 'LoadedJump':
                bm_vertjump[idx] = body_mass
                if db.loc[idx,'TestType'] =='' or type(db.loc[idx,'TestType'])!=str:
                    db.loc[idx,'TestType'] = test_type
                else:
                    db.loc[idx,'TestType'] = ', '.join(sorted([str(db.loc[idx,'TestType']), test_type]))# str(db.loc[idx,'TestType']) + ', ' + test_type
                execution_types = ['elastodyn', 'statodyn']
                for j,jump in enumerate(['CMJ', 'SJ']):# j,jump=0,'CMJ'
                    for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'P1/3', 'Fpos', 't_Fmax', 'tacc', 'tneg']:# par='Pmax'
                        for ld in [str(x) for x in [0, 20, 40, 60, 80, 100]]:# ld=str(0)
                            if '_'.join([jump,par,ld]) not in df.columns or np.isnan(db.loc[idx,'_'.join([jump,par,ld])]):# if there is already a value there from an 'Einzelsprung'
                                db.loc[idx,'_'.join([jump,par,ld])] = df[
                                    ((df['Ausführung']==execution_types[j]) & (90+float(ld) < df['%KG [%]']) & (df['%KG [%]'] < float(ld)+110))
                                    ][
                                        [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                                        ].mean()
                if not db.loc[idx][['_'.join([jump, 'Pmax_0']) for jump in ['CMJ','SJ']]].isna().any():
                    for par in ['Pmax', 's_max']:# par='Pmax'
                        for ld in [str(x) for x in [0, 20, 40, 60, 80, 100]]:# ld=str(0)
                            db.loc[idx, '_'.join(['CMJ', par, ld, 'effect_of_prestretch'])] = 100*(db.loc[idx, '_'.join(['CMJ', par, '0'])] / db.loc[idx, '_'.join(['SJ', par, '0'])] - 1)
             
            elif test_type == 'Einzelsprung':
                bm_vertjump[idx] = body_mass
                if db.loc[idx,'TestType'] =='' or type(db.loc[idx,'TestType'])!=str:
                    db.loc[idx,'TestType'] = test_type
                else:
                    db.loc[idx,'TestType'] = ', '.join(sorted([str(db.loc[idx,'TestType']), test_type]))# str(db.loc[idx,'TestType']) + ', ' + test_type
                execution_types = ['elastodyn', 'einbeinig links', 'einbeinig rechts']
                for jump in ['CMJ']:# jump='CMJ'
                    for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'P1/3', 'Fpos', 't_Fmax', 'tacc', 'tneg']:# par='Pmax'
                        for s,side in enumerate(['0', 'left','right']):# s,side = 0,'0'
                            db.loc[idx,'_'.join([jump, par, side])] = df[
                                ((df['Ausführung']==execution_types[s]))
                                ][
                                    [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                                    ].mean()
                    if not db.loc[idx][['_'.join([jump, 'Pmax', side]) for side in ['0', 'left', 'right']]].isna().any():
                        for par in ['Pmax', 's_max']:
                            db.loc[idx, '_'.join([jump, par, '0_bilateral_deficit'])] = 100*(1 - (db.loc[idx,'_'.join([jump, par, '0'])] / (db.loc[idx,'_'.join([jump, par, 'left'])] + db.loc[idx,'_'.join([jump, par, 'right'])])))# calculate bilateral deficit
                            # db.loc[idx, '_'.join([jump, par, '0_bilateral_deficit'])] = 100*(((db.loc[idx,'_'.join([jump, par, 'left'])] + db.loc[idx,'_'.join([jump, par, 'right'])]) / db.loc[idx,'_'.join([jump, par, '0'])]) - 1)# calculate bilateral deficit
                    if not db.loc[idx][['_'.join([jump,'Pmax',side]) for side in ['left','right']]].isna().any():
                        for par in ['Pmax', 's_max']:
                            db.loc[idx, '_'.join([jump, par, '0_LR-imbalance'])] = 100*(1 - np.min([db.loc[idx,'_'.join([jump, par, 'left'])], db.loc[idx,'_'.join([jump, par, 'right'])]])/np.max([db.loc[idx,'_'.join([jump, par, 'left'])], db.loc[idx,'_'.join([jump, par, 'right'])]]))
                for jump in ['SJ']:# jump='CMJ'
                    for par in ['Pmax', 'Ppos', 's_max', 'load', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'P1/3', 'Fpos', 't_Fmax', 'tacc', 'tneg']:# par='Pmax'
                        for ld in ['0']:
                            db.loc[idx,'_'.join([jump, par, ld])] = df[
                                ((df['Ausführung']=='statodyn'))
                                ][
                                    [col for col in df.columns if col.startswith(par)][0]
                                    ].mean()
                if not db.loc[idx][['_'.join([jump, 'Pmax_0']) for jump in ['CMJ','SJ']]].isna().any():
                    for par in ['Pmax', 's_max']:
                        db.loc[idx, '_'.join(['CMJ', par, '0_effect_of_prestretch'])] = 100*(db.loc[idx, '_'.join(['CMJ', par, '0'])] / db.loc[idx, '_'.join(['SJ', par, '0'])] - 1)
                    
                # 'Pmax_CMJ_left', 'Pavg_CMJ_left', 'h_CMJ_left', 'load_CMJ_left', 'spos_CMJ_left', 'tpos_CMJ_left', 'Fmax_CMJ_left', 'vmax_CMJ_left', 'Fv0_CMJ_left', 'P1/3_CMJ_left', 'Pmax_CMJ_right', 'Pavg_CMJ_right', 'h_CMJ_right', 'load_CMJ_right', 'spos_CMJ_right', 'tpos_CMJ_right', 'Fmax_CMJ_right', 'vmax_CMJ_right', 'Fv0_CMJ_right', 'P1/3_CMJ_right'
            elif test_type == 'Drop Jump':
                bm_dropjump[idx] = body_mass
                if db.loc[idx,'TestType'] =='' or type(db.loc[idx,'TestType'])!=str:
                    db.loc[idx,'TestType'] = test_type
                else:
                    db.loc[idx,'TestType'] = ', '.join(sorted([str(db.loc[idx,'TestType']), test_type]))# str(db.loc[idx,'TestType']) + ', ' + test_type
                execution_types = ['reaktiv']
                for j,jump in enumerate(['DJ']):# j,jump = 0,'DJ'
                    for par in ['Reak1', 'Reak2']:# par = 'Reak1'
                        for dh in [str(int(x)) for x in df['Automatic (112)'].dropna().unique()]:# dh = str(20)
                            if Options['alt_output']:
                                db.loc[idx,'_'.join([jump,par,dh])] = df[
                                    ((df['Ausführung']==execution_types[j]) & (df['Automatic (112)']==int(dh)))
                                    ][
                                        [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                                        ].max()
                            else:
                                db.loc[idx,'_'.join([jump,par,dh])] = df[
                                    ((df['Ausführung']==execution_types[j]) & (df['Automatic (112)']==int(dh)))
                                    ][
                                        [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                                        ].max()
                    for par in ['s_max', 'tacc', 's_pos', 'tpos', 'Fmax', 'Vmax', 'Fv0', 'Fpos', 't_Fmax', 'tacc', 'tneg']:# corresponding parameters for the trial where Reak1 was highest
                        for dh in [str(int(x)) for x in df['Automatic (112)'].dropna().unique()]:# dh = str(20)
                            # if Options['alt_output']:
                            #     # db.loc[idx,'_'.join([jump,par,dh])] = df[
                            #     #     ((df['Ausführung']==execution_types[j]) & (df['Automatic (112)']==int(dh)))
                            #     #     ][
                            #     #         [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                            #     #         ].max()
                            #     db.loc[idx,'_'.join([jump,par,dh])] = df.loc[
                            #         df[
                            #             ((df['Ausführung']==execution_types[j]) & (df['Automatic (112)']==int(dh)))
                            #             ][
                            #                 [col for col in df.columns if col.startswith('Reak1') and 'rel' not in col][0]
                            #                 ].idxmax(), [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                            #         ]
                            # else:
                            i = df[
                                ((df['Ausführung']==execution_types[j]) & (df['Automatic (112)']==int(dh)))
                                ][
                                    'Reak1:smax/t [cm/s*10]'
                                    ].idxmax()
                            db.loc[idx,'_'.join([jump,par,dh])] = df.loc[
                                i,
                                [col for col in df.columns if col.startswith(par) and 'rel' not in col][0]
                                ]
                            
    for col in all_columns:
        if col not in db.columns:
            db.insert(len(db.columns), col, np.nan)
            
    db = db[all_columns]
    db_rel = db.copy()
    for par in ['Fmax_70', 'Fmax_100']:
        for col in db_rel.columns:
            if col.startswith(par) and not any([p in col for p in ['bilateral_deficit', 'LR-imbalance']]):
                for i in db_rel.index:
                    try:# only works if the iso test was performed
                        db_rel.loc[i,col] = db.loc[i,col] / bm_iso[i]
                    except:
                        pass
    for par in ['_Pmax_', '_Ppos_', '_Fmax_', '_Fv0_', '_P1/3_']:
        for col in db_rel.columns:
            if par in col and not any([p in col for p in ['effect_of_prestretch', 'bilateral_deficit', 'LR-imbalance']]):
                for i in db_rel.index:
                    try:# works if either vertical jump test was performed
                        db_rel.loc[i,col] = db.loc[i,col] / bm_vertjump[i]
                    except:
                        pass
        
    if Options['alt_output']:
        db_alt = db_rel[
            [
                'TestDate', 'AthleteName', 'BirthDate', 'BodyMass',
                'CMJ_Pmax_0', 'SJ_Pmax_0', 'CMJ_Pmax_right', 'CMJ_Pmax_left', 'CMJ_Pmax_0_LR-imbalance',
                'CMJ_s_max_0', 'SJ_s_max_0', 'CMJ_s_max_right', 'CMJ_s_max_left', 'CMJ_Pmax_0_bilateral_deficit', 'CMJ_s_max_0_effect_of_prestretch', 'CMJ_s_pos_0',
                'DJ_Reak2_60', 'DJ_Reak1_60', 'DJ_s_max_60', 'DJ_tacc_60',
                'Fmax_100_bilateral', 'Fmax_100_left', 'Fmax_100_right', 'Fmax_100_LR-imbalance', 'Fmax_100_bilateral_deficit',
                'CMJ_Pmax_40', 'CMJ_s_max_40', 'CMJ_s_pos_40',
                'SJ_Pmax_40', 'SJ_s_max_40', 'SJ_s_pos_40',
                'DJ_Reak2_40', 'DJ_Reak1_40', 'DJ_s_max_40', 'DJ_tacc_40',
                'DJ_Reak2_20', 'DJ_Reak1_20', 'DJ_s_max_20', 'DJ_tacc_20',
                ]
            ]
        db_alt.insert(0, 'first_name', [str(db_alt.loc[i, 'AthleteName']).split(' ')[0] for i in db_alt.index])
        db_alt.insert(0, 'last_name', [str(db_alt.loc[i, 'AthleteName']).split(' ')[1] for i in db_alt.index])
        for side in ['bilateral', 'left', 'right']:
            db_alt.insert(0, 'Fmax_abs_100_'+side,
                          db['Fmax_100_'+side])
        db_alt.insert(0, 'Fmax_100_oneLeg',
                      db_rel[['Fmax_100_'+side for side in ['left', 'right']]].mean(axis=1)
                      )
        db_alt.insert(0, 'Fmax_abs_100_oneLeg',
                      db[['Fmax_100_'+side for side in ['left', 'right']]].mean(axis=1)
                      )
        for c in ['Nummer', 'Kader', 'Bemerkung', 'Groesse', 'Pmax_blank', 'Hupf', 'Laufsprung', 'Lateralsprung_l', 'Lateralsprung_r', 'Fmax_Einstellung', 'Standweitsprung']:
            db_alt.insert(0, c, '')

        db_alt = db_alt[
            [
                'Nummer',
                'TestDate', 'last_name', 'first_name', 'Kader', 'Bemerkung', 'BirthDate', 'Groesse', 'BodyMass',
                'CMJ_Pmax_0', 'SJ_Pmax_0', 'CMJ_Pmax_right', 'CMJ_Pmax_left', 'CMJ_Pmax_0_LR-imbalance',
                'CMJ_s_max_0', 'SJ_s_max_0', 'CMJ_s_max_right', 'CMJ_s_max_left', 'CMJ_Pmax_0_bilateral_deficit', 'CMJ_s_max_0_effect_of_prestretch', 'CMJ_s_pos_0',
                'DJ_Reak2_60', 'DJ_Reak1_60', 'DJ_s_max_60', 'DJ_tacc_60',
                'Pmax_blank',
                'Hupf', 'Laufsprung', 'Lateralsprung_l', 'Lateralsprung_r',
                'Fmax_abs_100_bilateral', 'Fmax_100_bilateral', 'Fmax_abs_100_left', 'Fmax_100_left', 'Fmax_abs_100_right', 'Fmax_100_right',
                'Fmax_Einstellung',
                'Fmax_abs_100_oneLeg', 'Fmax_100_oneLeg',
                'Fmax_100_LR-imbalance', 'Fmax_100_bilateral_deficit',
                'CMJ_Pmax_40', 'CMJ_s_max_40', 'CMJ_s_pos_40',
                'SJ_Pmax_40', 'SJ_s_max_40', 'SJ_s_pos_40',
                'Standweitsprung',
                'DJ_Reak2_40', 'DJ_Reak1_40', 'DJ_s_max_40', 'DJ_tacc_40',
                'DJ_Reak2_20', 'DJ_Reak1_20', 'DJ_s_max_20', 'DJ_tacc_20',
                ]
            ]
        # st.markdown(
        #     generate_excel_alt(db_alt),
        #     unsafe_allow_html=True
        #     )#, sign_digits=3
        return {'db_alt': db_alt}
    else:
        # st.markdown(
        #     generate_excel(db, db_rel),
        #     unsafe_allow_html=True
        #     )#, sign_digits=3
        return {'db': db, 'db_rel': db_rel}

#%%
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.write("""
    
    # Web app for transforming MLD csv-export files to excel database format
    
    """)
    st.sidebar.header('Options')
    Options = {}
    # Options = {'save_variables': False}# if '/app/' in os.getcwd() else True
    Options['valid_only'] = st.sidebar.checkbox('valid trials only',
                                                value=True,
                                                )
    Options['alt_output'] = st.sidebar.checkbox('alternative output format',
                                                value=False,
                                                )
    data_export_files = st.file_uploader("upload csv export file", accept_multiple_files=True)
    current_file_container = st.empty()
    if data_export_files is not None and len(data_export_files)>0:
        # if Options['save_variables']:
        #     for (path, _, files) in os.walk(os.path.join(os.getcwd(), 'saved_variables')):
        #         for f in files:
        #             os.remove(os.path.join(path, f))
        #     with open(os.path.join(os.getcwd(), 'saved_variables','Options.json'), 'w') as fp:
        #         json.dump(Options, fp)
        st.markdown(
            generate_excel(
                **process_uploaded_files(data_export_files, Options)
                ),
            unsafe_allow_html=True
            )
        


