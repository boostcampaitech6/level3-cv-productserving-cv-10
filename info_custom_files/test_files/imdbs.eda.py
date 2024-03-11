import numpy as np

# .npy 파일 로드
data = np.load('/home/ges/level3-cv-productserving-cv-10/data/sp-vqa/ibmb/new_imdb_train.npy', allow_pickle=True)

# 데이터 타입 출력
print(f"Data type: {type(data)}")

# 데이터의 전체 구조 확인 (예: shape, size)
print(f"Data shape: {data.shape}")
print(f"Data size: {data.size}")

# 데이터의 첫 번째 요소 타입 확인
print(f"First element type: {type(data[0])}")

# 데이터의 첫 번째 요소 내용 확인 (예시로, 배열이나 리스트일 경우)
if isinstance(data[0], (np.ndarray, list)):
    print(f"First element content: {data[0]}")

# 더 구체적인 데이터 구조 및 라벨링 확인
# 예: 첫 번째 요소가 딕셔너리라면, 키를 확인하여 어떤 라벨/필드가 있는지 확인
if isinstance(data[1], dict):
    print("Keys in the first dictionary element:", data[0].keys())
    
    

# for item in data :
#     print(item.keys())
# if isinstance(data[0], dict):
#     print("Keys in the first dictionary element:", data[0].values())
data[1].keys()
print(data[1])
import json