from datetime import date
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import os
from sklearn.preprocessing import RobustScaler as robust

with open('uber_price_pred_model.pkl', 'rb') as fr:
    model = pickle.load(fr)

def run_uber_ML():
    passenger_count = st.number_input("Jumlah Penumpang", 1, 6, key='1')
    distance_km = st.number_input("Jarak perjalanan", 2, 999, key='2')
    tanggal = st.date_input('Tanggal Pick up', value = date.today())
    waktu = st.time_input('Waktu Pick up')
        

    inputs = [[passenger_count, distance_km, waktu.hour, 
               waktu.minute, tanggal.day, tanggal.month, tanggal.year]]

    if st.button("Prediksi tarif"):
        result = model.predict(inputs)
        updated_res = result.flatten().astype(float)
        st.success('Tarifnya sebesar USD: {}'.format(updated_res))

if __name__ =='__main__':
    run_uber_ML()
