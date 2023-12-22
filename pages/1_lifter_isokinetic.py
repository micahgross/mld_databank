import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_excel(csv):
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    with pd.ExcelWriter(output, date_format='dd.mm.yyyy') as writer:
        csv.to_excel(writer,sheet_name='data',index=False)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc
    download_filename = 'lifter_data_isokinetic.xlsx'
    link_text = 'Download data'
    # return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download Excel databank</a>' # decode b'abc' => abc
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">{link_text}</a>' # decode b'abc' => abc



st.title('lifter isokinetic')
upload_file = st.file_uploader("upload csv export file", accept_multiple_files=False)
# csv = pd.read_csv(
#     os.path.join(os.path.split(os.getcwd())[0], 'data_transfer', 'isokiroh.csv'),
#     sep='\t',
#     skiprows=4,
#     header=None
#     ).iloc[:,:-1].dropna(axis='index').reset_index(drop=True).transpose()
if upload_file is not None:
    csv = pd.read_csv(
        upload_file,
        sep='\t',
        skiprows=4,
        header=None
        ).iloc[:,:-1].dropna(axis='index').reset_index(drop=True).transpose()
    n_reps = int(len(csv.columns) / 6)
    csv.columns = [
        '_'.join([str(rep+1), 'F', side, direction]) for rep in range(n_reps) for side in ['Links', 'Rechts', 'L+R'] for direction in ['auf', 'ab']
        ]
    st.write(csv)
    auf_fig = plt.figure()
    plt.title('auf')
    for col in csv.columns[4:][::6]:
        csv[col].plot(label=col)
    plt.legend() 
    st.pyplot(auf_fig)
    
    ab_fig = plt.figure()
    plt.title('ab')
    for col in csv.columns[5:][::6]:
        csv[col].plot(label=col)
    plt.legend()    
    st.pyplot(ab_fig)

    st.markdown(
        generate_excel(csv),
        unsafe_allow_html=True
        )#, sign_digits=3
