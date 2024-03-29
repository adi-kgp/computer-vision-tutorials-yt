from deepface import DeepFace
import json

# face matching
result = DeepFace.verify(img1_path='image_5.jpg', img2_path='image_8.jpg')
print(json.dumps(result, indent=2))


# find face in database
dfs = DeepFace.find(img_path="image_7.jpg", db_path="db")
print(dfs)


# face analysis
objs = DeepFace.analyze(img_path="eminem.jpg", enforce_detection=False)
print(json.dumps(objs, indent=2))

