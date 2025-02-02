import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import re

# Load the model and tokenizer
@st.cache_resource
def load_components():
    model = load_model('auto_correction_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_components()
max_sequence_length = 394  

# Text cleaning functions
arabic_punctuations = [".","#","$","//","?","=","'","_","-","';","\\","`","؛","<",">","(",")","*","&","^","%","]","[",",","ـ","،","/",":","؟","{","}","~","|","!","”","…","“","–"]

def clean_text(text):
    text = re.sub('[{}]'.format(re.escape(''.join(arabic_punctuations))), '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("\d+", " ", text)
    return text.strip()

# Prediction function
def correct_sentence(input_text):
    cleaned_text = clean_text(input_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    predictions = model.predict(padded)
    
    # Get the predicted sequence and convert it back to text
    corrected_sequence = np.argmax(predictions, axis=-1)[0]
    corrected_text = tokenizer.sequences_to_texts([corrected_sequence])[0]
    
    return corrected_text  # لا حاجة لاستخدام get_display أو arabic_reshaper

st.title('التصحيح التلقائي للنصوص العربية')
st.header('تصحيح الأخطاء الإملائية باستخدام الذكاء الاصطناعي')

input_text = st.text_area("أدخل النص الذي تريد تصحيحه:", height=150)

if st.button('تصحيح النص'):
    if input_text:
        corrected_text = correct_sentence(input_text)
        
        st.subheader("النص المدخل:")
        st.markdown(f'<div style="text-align: right; direction: rtl; font-size: 18px;">{input_text}</div>', unsafe_allow_html=True)
        
        st.subheader("النص المصحح:")
        st.markdown(f'<div style="text-align: right; direction: rtl; color: green; font-size: 18px;">{corrected_text}</div>', unsafe_allow_html=True)
    else:
        st.warning("الرجاء إدخال نص لتصحيحه")

st.markdown("""
### كيفية الاستخدام:
1. أدخل النص العربي الذي يحتوي على أخطاء في المنطقة النصية
2. انقر على زر 'تصحيح النص'
3. سيظهر النص المصحح في المنطقة الخضراء أدناه

### ملاحظات:
- النموذج مدرب على تصحيح الأخطاء الإملائية الشائعة
- قد لا يكون التصحيح دقيقًا بنسبة 100% في جميع الحالات
- يدعم النموذج النصوص الطويلة حتى 400 كلمة تقريبًا
""")
