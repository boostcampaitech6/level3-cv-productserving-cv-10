# Infographics VQA - pix2struct


🚀 **[모델 시연 바로가기](https://app-test-hapfsdaay6yfrirfdoagqg.streamlit.app/)** 🚀

InfographicsVQA는 DocVQA task의 한 종류로서 Document의 내용에 대한 질의 응답을 하는 **multi-modal AI**입니다.

특히, Documentation내부에 Infographic한 data[table, figure, map등]가 많이 포함돼있는 경우에 특화된 모델입니다.

Backbone model로서 **Pix2Struct**를 활용하였고 아래는 기존 모델들과 다른 특징입니다.

### 1. ViT를 사용하지만 patch를 만들때 sqaure로 resize하지 않고 input이미지의 ratio(종횡비)를 유지하며 만듭니다.
### 2. Question(text data)를 image의 top에 rendering하여 학습 데이터로 활용합니다.
### 3. pre-pretrain 전략으로서 masked parsing HTML이미지를 학습합니다.
     이는 layout을 토대로 image와 text를 정확하게 1:1매칭이 되는 DocVQA 전반에 활용가능한 매우 적합한 데이터셋입니다.


위 시연 페이지를 통해 feature extract가 어려운 이미지에서도 잘 작동하는 모델의 성능을 테스트할 수 있습니다.

## 사용 방법

1. 위의 링크를 클릭하여 웹앱에 접속합니다.
2. 예시 이미지를 이용하거나 새로운 infographics 이미지를 업로드 합니다.
3. 이미지에서 추출할 수 있는 question을 입력합니다.
4. [get answer]을 버튼을 통해 정답을 추출할 수 있습니다.
