from flask import Flask, request, render_template
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 사용자 입력 받기 (예: 파일 업로드)
        # file = request.files['file']
        # file_path = 'path/to/save/' + file.filename
        # file.save(file_path)
        
        # TorchServe에 추론 요청 보내기
        response = requests.post('http://localhost:8080/predictions/my_model', files={'data': open(file_path, 'rb')})
        result = response.text
        
        # 결과를 웹 페이지에 표시
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
