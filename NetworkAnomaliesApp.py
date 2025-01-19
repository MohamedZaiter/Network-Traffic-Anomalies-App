import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt


uploaded_file = st.file_uploader("choisir un fichier csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write(data.head())
    
    if 'Label' in data.columns:  
        label_encoder = LabelEncoder()
        data['Label'] = label_encoder.fit_transform(data['Label'])
    
    data = pd.get_dummies(data, drop_first=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True) 
    data.fillna(data.mean(), inplace=True) 
    
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_scaled = scaler.fit_transform(data[numeric_columns])

    pca = PCA(n_components=10)
    pca_components = pca.fit_transform(data_scaled)

    pca_df = pd.DataFrame(data=pca_components, columns=[f'PC{i+1}' for i in range(10)])

    st.write("PCA Components:")
    st.write(pca_df.head())

   
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    
    
    iso_forest.fit(pca_df)  
    
    predictions = iso_forest.predict(pca_df)

    pca_df['Anomaly'] = predictions

    st.write("Anomalies Detected:")
    st.write(pca_df[pca_df['Anomaly'] == -1])  

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Anomaly'], cmap='coolwarm', label='Normal')
    plt.title('Anomalies Detected by Isolation Forest')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    st.pyplot(plt)
