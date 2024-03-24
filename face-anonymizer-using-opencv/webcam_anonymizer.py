import cv2
import mediapipe as mp # using mediapipe face detector to detect faces
import os
import argparse


def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    H, W, _ = img.shape
    
    #print(out.detections)
    
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

    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam') # image or video or webcam
args.add_argument("--filePath", default=None)

args = args.parse_args()


# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    
    
    if args.mode in ['image']:
        # read image
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
            
        # save image
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cv2.imwrite(os.path.join(output_dir, 'img_face_blurred.png'), img)


    elif args.mode in ['video']:
        # read video
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        
        output_dir = './output'
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'face_blur_video.mp4'),
                                       fourcc = cv2.VideoWriter_fourcc(*'MP4V'),
                                       fps=25,
                                       frameSize=(frame.shape[1], frame.shape[0])
                                       )
        
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()
            
        cap.release()
        output_video.release()


    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            ret, frame = cap.read()
        
        cap.release()