import cv2, time

# load model
model_path = 'opencv_face_detector_uint8.pb'
config_path = 'opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

# initialize video source, default 0 (webcam)
video_path = 'txt.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output_opencv_dnn.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count, tt = 0, 0

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    print('no frame')
    break
  frame_count += 1

  start_time = time.time()

  # prepare input
  result_img = img.copy()
  h, w, _ = result_img.shape
  blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
  net.setInput(blob)

  # inference, find faces
  detections = net.forward()

  maxX1 = maxY1 = maxX2 = maxY2 = maxW = maxH = 0
  # postprocessing
  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
      x1 = int(detections[0, 0, i, 3] * w)
      y1 = int(detections[0, 0, i, 4] * h)
      x2 = int(detections[0, 0, i, 5] * w)
      y2 = int(detections[0, 0, i, 6] * h)


      face_img = result_img[y1:y2, x1:x2]  # 인식된 얼굴 영역 이미지 face_img에 저장

      if ((x2 - x1) * (y2 - y1) > maxW * maxH):
        maxX1 = x1
        maxY1 = y1
        maxX2 = x2
        maxY2 = y2
        maxW = x2 - x1
        maxH = y2 - y1

      if ((maxX1 != x1 and maxX2 != x2 and maxY1 != y1 and maxY2 != y2 and maxW != (x2-x1) and maxH != (y2-y1))
      and (confidence * 100.)>40):
        face_img = cv2.resize(face_img, None, fx=0.04, fy=0.04)  # 축소
        face_img = cv2.resize(face_img, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)  # 확대
        result_img[y1:y2, x1:x2] = face_img  # 인식된 얼굴 영역 모자이크 처리

  # inference time
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, 'FPS(dnn): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # visualize
  cv2.imshow('result', result_img)
  if cv2.waitKey(1) == ord('q'):
    break

  out.write(result_img)

cap.release()
out.release()
cv2.destroyAllWindows()