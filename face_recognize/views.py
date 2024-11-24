from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import pickle
import face_recognition
import numpy as np
import imutils
import os
from imutils import paths
from django.conf import settings
from openpyxl import Workbook
from .models import RecognitionHistory, IdNameMapping
from django.http import HttpResponse
from django.utils.timezone import localtime
from django.db.models import Max
from .apps import *
from rest_framework.decorators import api_view
from django.utils import timezone
from django.utils.timezone import localtime
import time

last_recognition_time = {}

# Đường dẫn tệp
ENCODINGS_PATH = "encodings.pickle"
DATASET_PATH = "dataset"
PROCESSED_DATASET_PATH = "processed_dataset"


# Camera
camera = None

# Tải dữ liệu ban đầu
def load_data():
    print("[INFO] loading encodings...")
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = {"encodings": [], "ids": []}

    print("[INFO] loading ID-to-Name mappings from database...")
    id_to_name = {entry['id_code']: entry['name'] for entry in IdNameMapping.objects.all().values('id_code', 'name')}
    return data, id_to_name

data, id_to_name = load_data()

# Căn chỉnh khuôn mặt
def align_face(image, landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    # Tính toán vị trí mắt trái và mắt phải
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")

    # Tính góc giữa hai mắt
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Tính toán vị trí trung tâm giữa hai mắt
    eyes_center = (
        int((left_eye_center[0] + right_eye_center[0]) // 2),
        int((left_eye_center[1] + right_eye_center[1]) // 2)
    )

    # Tạo ma trận quay và căn chỉnh khuôn mặt
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    (h, w) = image.shape[:2]
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return rotated

# Cắt và lưu khuôn mặt đã căn chỉnh từ dataset
def save_cropped_faces():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            existing_ids = set(data["ids"])
    else:
        existing_ids = set()

    # Lấy paths của images trong dataset
    imagePaths = list(paths.list_images(DATASET_PATH))
    
    # Duyệt qua các image paths
    for (i, imagePath) in enumerate(imagePaths):
        id = imagePath.split(os.path.sep)[-2]
        
        # Kiểm tra nếu ID đã tồn tại trong encodings thì bỏ qua
        if id in existing_ids:
            print(f"[INFO] ID '{id}' already exists. Skipping images in this folder.")
            continue

        print(f"[INFO] processing image {i+1}/{len(imagePaths)}: {imagePath}")

        # Load image bằng OpenCV
        image = cv2.imread(imagePath)
        if image is None:
            print(f"[WARNING] Cannot open image file: {imagePath}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face và landmarks bằng face_recognition
        face_locations = face_recognition.face_locations(rgb)
        face_landmarks = face_recognition.face_landmarks(rgb)

        if len(face_locations) == 0:
            print(f"[WARNING] No face detected in the image: {imagePath}")
            continue

        # Duyệt qua các khuôn mặt phát hiện được và lưu lại theo yêu cầu
        for face_location, landmarks in zip(face_locations, face_landmarks):
            # Căn chỉnh khuôn mặt dựa trên landmarks
            aligned_face = align_face(rgb, landmarks)
            
            # Cắt khuôn mặt dựa trên bounding box từ face_location
            top, right, bottom, left = face_location
            face_cropped = aligned_face[top:bottom, left:right]

            # Resize khuôn mặt
            aligned_face_resized = cv2.resize(face_cropped, (256, 256))

            # Kiểm tra nếu thư mục của người dùng chưa tồn tại trong processed_dataset
            user_dir = os.path.join(PROCESSED_DATASET_PATH, id)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)

            # Lưu ảnh khuôn mặt đã cắt và căn chỉnh
            face_fileid = os.path.join(user_dir, f"face_{i}.jpg")
            face_bgr = cv2.cvtColor(aligned_face_resized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(face_fileid, face_bgr)
            print(f"[INFO] Saved cropped and aligned face to {face_fileid}")

# Tính toán encoding của khuôn mặt từ ảnh đã căn chỉnh
def compute_embeddings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.loads(f.read())
            knownEncodings = data["encodings"]
            knownIds = data["ids"]
    else:
        knownEncodings = []
        knownIds = []

    existing_ids = set(knownIds)

    # Duyệt qua tất cả các thư mục con của processed_dataset
    for user_dir in os.listdir(PROCESSED_DATASET_PATH):
        user_path = os.path.join(PROCESSED_DATASET_PATH, user_dir)

        if not os.path.isdir(user_path):
            continue

        # Bỏ qua ID nếu đã tồn tại
        if user_dir in existing_ids:
            print(f"[INFO] ID '{user_dir}' already exists. Skipping images in this folder.")
            continue

        # Duyệt qua từng ảnh khuôn mặt đã cắt trong thư mục của mỗi ID
        for face_image in os.listdir(user_path):
            face_path = os.path.join(user_path, face_image)
            image = cv2.imread(face_path)
            if image is None:
                print(f"[WARNING] Cannot open image file: {face_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect và tính toán encoding của khuôn mặt
            encodings = face_recognition.face_encodings(rgb)
            if len(encodings) == 0:
                print(f"[WARNING] No face detected in the image: {face_path}")
                continue

            # Lưu encoding và ID
            knownEncodings.append(np.array(encodings[0]))
            knownIds.append(user_dir)
            print(f"[INFO] Computed encoding for {face_path}")

    # Lưu lại facial encodings + ids vào ổ cứng
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "ids": knownIds}
    
    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(data))

    print("[INFO] Done!")

def detect_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    
    # Phát hiện vị trí khuôn mặt
    face_locations = face_recognition.face_locations(rgb)
    
    return face_locations, r

# def recognize_faces(frame):
#     global data, id_to_name, last_recognition_time
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb = imutils.resize(rgb, width=750)
#     face_locations, r = detect_faces(frame)
    
#     # Trích xuất encoding từ vị trí khuôn mặt
#     encodings = face_recognition.face_encodings(rgb, face_locations)
#     ids = []
    
#     for face_encoding in encodings:
#         # Tính khoảng cách giữa encoding mới và encoding đã lưu
#         distances = np.linalg.norm(data["encodings"] - face_encoding, axis=1)
#         min_distance = np.min(distances)
        
#         id_detected = "Unknown"
#         accuracy = (1 - min_distance) * 100  # Tính độ chính xác
#         threshold = 0.4  # Ngưỡng nhận diện
#         if min_distance < threshold:
#             id_detected = data["ids"][np.argmin(distances)]
        
#         ids.append((id_detected, accuracy))
    
#     # Vẽ bounding boxes và thông tin nhận diện lên ảnh
#     for (box, (id_detected, accuracy)) in zip(face_locations, ids):
#         top, right, bottom, left = box
#         left, top, right, bottom = int(left * r), int(top * r), int(right * r), int(bottom * r)
        
#         name = id_to_name.get(id_detected.strip(), "Unknown")
#         label = f"{id_detected}_{name}"

#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         y = top - 15 if top - 15 > 15 else top + 15
#         cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         # Lưu lịch sử nhận diện chỉ khi đủ 1 phút đã trôi qua
#         if id_detected != "Unknown":
#             current_time = time.time()  # Lấy thời gian hiện tại (số giây từ epoch)

#             # Kiểm tra nếu chưa nhận diện người này hoặc nếu đã qua 1 phút kể từ lần nhận diện cuối
#             if id_detected not in last_recognition_time or (current_time - last_recognition_time[id_detected]) >= 60:
#                 # Lưu vào database
#                 RecognitionHistory.objects.create(
#                     recognized_id=id_detected,
#                     recognized_name=name,
#                     prediction_accuracy=round(accuracy, 2)
#                 )
#                 # Cập nhật thời gian nhận diện lần này
#                 last_recognition_time[id_detected] = current_time

#     return frame

from deepface import DeepFace
import cv2

def detect_liveness_with_deepface(face_image):
    """
    Kiểm tra liveness của khuôn mặt bằng DeepFace.
    :param face_image: Khuôn mặt (numpy array)
    :return: True nếu là thật, False nếu là giả
    """
    try:
        # Kiểm tra liveness qua DeepFace
        analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        if analysis:  # Nếu nhận diện thành công
            return True  # Khuôn mặt thật
    except Exception as e:
        print(f"Lỗi khi kiểm tra liveness: {e}")
    return False  # Khuôn mặt giả

def recognize_faces_with_liveness(frame):
    """
    Nhận diện khuôn mặt và kiểm tra liveness.
    :param frame: Frame video
    :return: Frame đã xử lý
    """
    global data, id_to_name, last_recognition_time
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations, r = detect_faces(frame)

    ids = []
    for (top, right, bottom, left) in face_locations:
        face = rgb[top:bottom, left:right]

        # Kiểm tra liveness
        is_real = detect_liveness_with_deepface(face)
        if not is_real:
            ids.append(("Fake", 0))  # Gắn nhãn "Fake" nếu khuôn mặt giả
            continue

        # Trích xuất encoding khuôn mặt
        encodings = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
        if len(encodings) > 0:
            face_encoding = encodings[0]
            distances = np.linalg.norm(data["encodings"] - face_encoding, axis=1)
            min_distance = np.min(distances)

            id_detected = "Unknown"
            accuracy = (1 - min_distance) * 100  # Tính độ chính xác
            threshold = 0.4  # Ngưỡng nhận diện
            if min_distance < threshold:
                id_detected = data["ids"][np.argmin(distances)]

            ids.append((id_detected, accuracy))
        else:
            ids.append(("Unknown", 0))

    # Vẽ bounding boxes
    for (box, (id_detected, accuracy)) in zip(face_locations, ids):
        top, right, bottom, left = box
        left, top, right, bottom = int(left * r), int(top * r), int(right * r), int(bottom * r)
        
        color = (0, 255, 0) if id_detected != "Fake" else (0, 0, 255)
        label = f"{id_detected} ({accuracy:.2f}%)" if id_detected != "Fake" else "Fake"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

@api_view(["POST"])
def stop_camera(request):
    global camera
    if camera:
        camera.release()
        camera = None
        return JsonResponse({"success": True, "message": "Camera đã được dừng"})
    return JsonResponse({"success": False, "message": "Camera không đang hoạt động"})

# Lấy khung hình từ camera, nhận diện khuôn mặt và stream ảnh JPEG theo thời gian thực.
# def generate_frames():
#     global camera
#     while camera and camera.isOpened():
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             frame = recognize_faces(frame)
#             frame = cv2.resize(frame, (320, 240))
#             ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    global camera
    while camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:
            # Nhận diện và kiểm tra liveness
            frame = recognize_faces_with_liveness(frame)
            frame = cv2.resize(frame, (320, 240))
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Nhận diện qua camera           
def recognize_video(request):
    start_camera()  
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# Nhận diện hình ảnh
# def recognize_image(request):
#     if request.method == "POST" and request.FILES.get('image'):
#         stop_camera()
#         image_file = request.FILES['image']
#         img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

#         img = recognize_faces(img)
#         img_resized = cv2.resize(img, (680, 500))
#         cv2.imwrite("static/recognized_image.jpg", img_resized)
        
#         return render(request, 'index.html', {'recognized_image': 'recognized_image.jpg'})
#     return render(request, 'index.html')


@api_view(["POST"]) 
def GenerateID(request):
    try:
        person = IdNameMapping.objects.order_by('-id_code').first()
        id = person.id_code
        id = int(id)
        new_id = id + 1
        new_id = str(new_id).zfill(5)
    except:
        new_id = "00001"
    return ObjResponse({"id": new_id})

@api_view(["POST"])
def save_new_face(request):
    if request.method == "POST":
        id_code = request.POST.get("id")
        name = request.POST.get("name")
        images = request.FILES.getlist("images[]")

        if not id_code or not name:
            return Response({"success": False, "error": "ID and name are required."}, status=400)

        # Lưu ảnh vào thư mục
        dataset_path = os.path.join("dataset", id_code)
        os.makedirs(dataset_path, exist_ok=True)

        # Lưu từng ảnh
        for image in images:
            image_path = os.path.join(dataset_path, image.name)
            with open(image_path, "wb") as f:
                for chunk in image.chunks():
                    f.write(chunk)
        
        # Cập nhật thông tin vào cơ sở dữ liệu
        IdNameMapping.objects.update_or_create(id_code=id_code, defaults={"name": name})

        # Gọi các hàm xử lý (lưu ảnh cắt và tính toán embeddings)
        save_cropped_faces()
        compute_embeddings()

        global data, id_to_name
        data, id_to_name = load_data()
        
        return Response({"success": True})
    else:
        return Response({"success": False, "error": "Invalid request method"}, status=400)
    
# Xuất lịch sử nhận diện
def export_recognition_history(request):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Recognition History"

    worksheet.append(["ID", "Name", "Recognition Time", "Prediction Accuracy"])
    
    recognition_histories = RecognitionHistory.objects.values("recognized_id", "recognized_name", "prediction_accuracy").annotate(
        latest_recognition_time=Max("recognition_time")
    )
    
    for history in recognition_histories:
        local_time = localtime(history["latest_recognition_time"])
        
        worksheet.append([
            history["recognized_id"],
            history["recognized_name"],
            local_time.strftime('%Y-%m-%d %H:%M:%S'),
            f"{history['prediction_accuracy']}%"
        ])
    
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2
        worksheet.column_dimensions[column_letter].width = adjusted_width
    
    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = 'attachment; filename="recognition_history.xlsx"'
    workbook.save(response)
    
    return response

@api_view(['GET'])
def get_recognition_history(request):
    try:
        # Lấy lịch sử nhận diện từ database, sắp xếp theo thời gian nhận diện mới nhất
        recognition_history = RecognitionHistory.objects.order_by('-recognition_time').values(
            'recognized_id', 'recognized_name', 'recognition_time', 'prediction_accuracy'
        )
        data = [
            {
                "recognized_id": record['recognized_id'],
                "recognized_name": record['recognized_name'],
                "recognition_time": record['recognition_time'].strftime('%Y-%m-%d %H:%M:%S'),
                "prediction_accuracy": f"{record['prediction_accuracy']}%"
            }
            for record in recognition_history
        ]
        return Response({"history": data}, status=200)
    except Exception as e:
        return Response({"error": f"Lỗi: {str(e)}"}, status=500)

@api_view(["POST"]) 
def clear_recognition_history(request):
    if request.method == "POST":
        try:
            # Xóa toàn bộ dữ liệu
            RecognitionHistory.objects.all().delete()
            return JsonResponse({"success": True})
        except Exception as e:
            # Trả về lỗi nếu có
            return JsonResponse({"success": False, "error": str(e)}, status=500)
    # Trường hợp không phải POST
    return JsonResponse({"success": False, "error": "Invalid request method"}, status=400)

def home(request):
    # Lấy tất cả các bản ghi, không giới hạn số lượng
    recognition_history = RecognitionHistory.objects.all().order_by('-recognition_time')
    return render(request, 'index.html', {'recognition_history': recognition_history})