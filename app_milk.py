 #Bibliotecas para manipulção de dados
import pandas as pd
import numpy as np
# importando streamlit
import streamlit as st

# Bibliotecas utilizadas na Construção de Máquinas Preditivas
import xgboost
from xgboost import XGBClassifier
# biblioteca para imoortação de modelo
from joblib import load

# função para carregar o dataset
@st.cache_data
def get_data():
    return pd.read_csv("milknew.csv")

# função para importar modelo preditivo
def import_model():
    return(load('modelo.joblib'))

def mapear_saida(valor):
     if valor == 0:
         return "high"
     elif valor == 1:
         return "low"
     elif valor == 2:
         return "medium"

data = get_data()

model = import_model()

# título
st.title("Prevendo a qualidade do leite")

# subtítulo
st.subheader("Você pode prever qualidade do leite inserindo suas propriedades ao lado")

st.sidebar.subheader("Insira os dados do leite")

# mapeando dados do usuário para cada atributo
pH = st.sidebar.number_input("Valor do pH")
temperatura = st.sidebar.number_input("Temperatura")
taste = st.sidebar.number_input("Gosto (0 ou 1) ")
odor= st.sidebar.number_input("Odor (0 ou 1)")
fat = st.sidebar.number_input("Gordura (0 ou 1)")
turbidity = st.sidebar.number_input("Turbidez (0 ou 1)")
colour = st.sidebar.number_input("Cor (código)")

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição da Qualidade")

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[pH , temperatura, taste, odor, fat, turbidity, colour]])
    st.subheader("A qualidade do leite é:")

    saida_mapeada = mapear_saida(result)
    st.write(saida_mapeada)

# verificando o dataset
st.subheader("Dados usados para o treinamento" )


# exibindo os top 8 registro do dataframe
st.dataframe(data.head(7))

