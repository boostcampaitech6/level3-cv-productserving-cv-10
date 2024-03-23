import streamlit as st
from PIL import Image
from inference import get_answer  # inference.py에서 get_answer 함수를 임포트

# Streamlit UI 구성
st.set_page_config(layout="wide")  # 전체 페이지를 wide 모드로 설정
st.title('Visual Question Answering System')
# 상태 초기화, answer 키가 없으면 빈 문자열로 초기화
if 'answer' not in st.session_state:
    st.session_state['answer'] = ""

# 이미지 업로드 및 질문/답변 섹션 나란히 배치
col1, col_, col2, col3 = st.columns([2, 0.5 , 2, 1], gap='small')  # 이미지 섹션(col1)과 질문/답변 섹션(col2) 배치, 그리고 오른쪽 빈 공간(col3)

with col1:
    st.header("Image")
    example_image_path = '10019.jpeg'
    # 이미지 업로드 버튼
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
    # 사용자가 이미지를 업로드한 경우 업로드한 이미지를 표시, 그렇지 않은 경우 예시 이미지를 표시
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        image = Image.open(example_image_path)
    st.image(image, width=300)  # 이미지 너비를 300px로 설정

with col2:
    st.header("Question")
    example_question = 'How much revenue in billions is expected from foreign spectators?'
    question = st.text_input("Enter your question about the image:",placeholder=example_question ,key="question")  # 질문 입력란
    # How much revenue in billions is expected from foreign spectators
    if st.button('Get Answer', key="answer_button"):
        if question:  # 이미지가 있든 없든, 질문이 있으면 정답을 찾음
            st.session_state['answer'] = get_answer(image, question)
        else:
            st.session_state['answer'] = "Please enter a question."
    st.header("Answer")
    st.text_input("Answer", value=st.session_state['answer'], key="answer_input")  # 답변 입력란

