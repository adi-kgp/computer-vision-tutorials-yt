import cv2
import mediapipe as mp # using mediapipe face detector to detect faces
import os

# read image

img_path = 'test_face.jpg'

img = cv2.imread(img_path)

H, W, _ = img.shape


# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    print(out.detections)
    
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h  = bbox.xmin, bbox.ymin, bbox.width, bbox.height
    
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], ksize=(50,50))
            
            
# save image
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
cv2.imwrite(os.path.join(output_dir, 'img_face_blurred.png'), img)

