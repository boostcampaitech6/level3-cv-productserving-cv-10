import streamlit as st
from PIL import Image
from inference import get_answer  # inference.py에서 get_answer 함수를 임포트
import streamlit as st
from PIL import Image

# Streamlit UI 구성
st.title('Visual Question Answering System')

# 조정된 컬럼 설정: 이미지 컬럼에 더 많은 공간 할당, 질문 컬럼 너비 확장
col1, col2 = st.columns([1.5, 3])  # 질문 컬럼(col2)이 이미지 컬럼(col1)보다 넓게 설정

with col1:
    st.header("Image")
    example_image_path = '/home/ges/level3-cv-productserving-cv-10/data/infographics/images/70538.jpeg'
    
    # 이미지 업로드 버튼
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
    
    # 사용자가 이미지를 업로드한 경우 업로드한 이미지를 표시
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    # 사용자가 이미지를 업로드하지 않은 경우 예시 이미지를 표시
    else:
        example_image = Image.open(example_image_path)
        st.image(example_image, use_column_width=True)

with col2:
    st.header("Question and Answer")
    example_question = "ex) What is the main color of the shirt?"
    question = st.text_input("Enter your question about the image:", placeholder=example_question, key="question")

    if st.button('Get Answer'):
        if uploaded_file is not None and question:
            # `get_answer` 함수를 호출하여 답변을 얻음
            answer = get_answer(image, question)  # 여기서 `image`는 업로드한 이미지 객체
            # answer = "Here will be the model's answer."  # 임시 답변, 실제 모델의 답변 로직에 따라 변경 필요
        else:
            answer = "Please upload an image and enter a question."
        st.write("Answer:", answer)
