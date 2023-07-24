import cv2
import numpy as np
import onnxruntime
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from mtcnn import MTCNN
from insightface.app.common import Face
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

get_embd=ArcFaceONNX(model_file='/home/adityat/Downloads/ms1mv2_r50_pfc.onnx')
dataset_path='/home/adityat/Downloads/Dataset'
def get_embedding(image_path):
	img=cv2.imread(image_path)
	detector=MTCNN()
	result=detector.detect_faces(img)
	print(result)
	if len(result)==0:
		print(result)
		d={}
		arr=np.empty((5,2))
		d['kps']=arr
		face=Face(d)
		return get_embd.get(img,face)
	elif 'keypoints' not in result[0]:
		print(result)
		d={}
		d['kps']=None
		face=Face(img, d)
		cv2.imshow(face)
		return get_embd.get(img, face)
	else:
		kps=result[0]['keypoints']
		print(kps)
		arr=list(kps.values())
		arr=np.array(arr)
		dic={}
		dic['kps']=arr
		f=Face(d=dic)
		return get_embd.get(img, f)

person_folders=os.listdir(dataset_path)
all_embeddings = []
all_names=[]
for person in person_folders:
        person_path = os.path.join(dataset_path, person)
        image_files = os.listdir(person_path)
        embeddings = []
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            embedding = get_embedding(image_path)
            embeddings.append(embedding.reshape(-1,2))


        if len(embeddings) > 0:
            embeddings = np.concatenate(embeddings, axis=0)
            all_embeddings.append(embeddings)
            all_names.append(person)

all_embeddings = np.concatenate(all_embeddings, axis=0)
cosine_matrix = cosine_distances(all_embeddings)

print("Cosine distance Matrix:")
print(cosine_matrix)

for i in range(len(all_names)):
        for j in range(i , len(all_names)):
            person1_name = all_names[i]
            person2_name = all_names[j]
            person1_idx = np.arange(i * 4, i * 4 + 4)
            person2_idx = np.arange(j * 4, j * 4 + 4)
            pairwise_similarity_matrix = cosine_matrix[person1_idx][:, person2_idx]
            print(f"{person1_name} and {person2_name}:")
            print(pairwise_similarity_matrix)
