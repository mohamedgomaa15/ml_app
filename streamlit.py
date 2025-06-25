import streamlit as st
import joblib 

model = joblib.load('linear_regression_model.pk1')
scaler = joblib.load('scaler.pk1')



st.title('Preidict Your Score')
st.write('Get your score based on #hours of Studing')

hours = st.number_input('Studing Hours', min_value=0.0, step=1.0)


if st.button('predict'):
    try:
        data = [[hours]]
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)
        st.write(f'Your prediction is= {pred[0]:.2f}')
    except Exception as e:
        st.error(f'Erorr: {e}')

