from flask import Flask, render_template, request, redirect, url_for, jsonify,send_file, send_from_directory, session, current_app, Response, make_response
from flask_redis import FlaskRedis
from flask_cors import CORS
import redis
# from gunicorn.app.base import Application
from werkzeug.utils import secure_filename
from werkzeug.local import LocalProxy, LocalStack
import psycopg2
import os,io,threading,time,random
import json
import PIL
from PIL import Image
import subprocess
import cv2.dnn
import numpy as np
from matplotlib import pyplot as plt
import base64

import torch
import ultralytics
from ultralytics import YOLO


ERRFILESIZE = "파일크기가 200MB보다 작아야합니다."
ERRFILEFORMAT = "가능한 파일 포멧은 png,jpg,mp4 입니다."

app = Flask(__name__)
app.secret_key = 'server_admin'
app.static_folder = 'static'
app.static_url_path = '/static'

# Flask-Session 설정
# app.config['SESSION_TYPE'] = 'filesystem'  # 세션을 파일 시스템에 저장
# app.config['SESSION_PERMANENT'] = False     # 세션을 브라우저를 닫으면 종료되도록 설정
# app.config['SESSION_USE_SIGNER'] = True      # 세션 데이터에 서명 사용 (선택 사항)
# Session(app)

app.config['REDIS_URL'] = 'redis://localhost:6379/0'
redis_client = FlaskRedis(app)

# Redis 연결 설정
redis_host = 'localhost'
redis_port = 6379
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

# flask-cors
CORS(app)

# @app.before_request
# def before_request():
#     if 'progress' not in session:
#         session['progress'] = 0

def reset_progress():  
    redis_client.set('progress', 0)

def set_progress(progress):
    redis_client.set('progress', progress)

def get_progress():
    return int(redis_client.get('progress'))

# class FlaskGunicornApp(Application):
#     def init(self, parser, opts, args):
#         pass

#     def load(self):
#         return app

def database_initialize(db_info):
    try:
        conn = psycopg2.connect(**db_info)
        cursor = conn.cursor()

        cursor.execute("""Drop TABLE IF EXISTS samplemodel_metadata_table CASCADE;""")
        cursor.execute("""Drop TABLE IF EXISTS model_metadata_table CASCADE;""")
        cursor.execute("""Drop TABLE IF EXISTS aihub_img_table CASCADE;""")
        cursor.execute("""Drop TABLE IF EXISTS aihub_annotation_table CASCADE;""")
        cursor.execute("""Drop TABLE IF EXISTS upload_data_table CASCADE;""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS samplemodel_metadata_table (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE,
                        size_mb FLOAT,
                        format VARCHAR(255)
                        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS model_metadata_table (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE,
                        size_mb FLOAT,
                        format VARCHAR(255)
                        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS aihub_img_table (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE,
                        size_mb FLOAT,
                        type VARCHAR(255),
                        split VARCHAR(255),
                        image BYTEA NOT NULL
                        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS aihub_annotation_table (
                        image_id INT REFERENCES aihub_img_table(id),
                        annotation_id SERIAL PRIMARY KEY , 
                        category_id INT,
                        category_str VARCHAR(255),
                        bboxs FLOAT[],
                        segmentation FLOAT[],
                        is_crowd INT
                        );""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS upload_data_table (
                        id SERIAL PRIMARY KEY,
                        file_name VARCHAR(255) UNIQUE,
                        size_mb FLOAT,
                        extension VARCHAR(255),
                        format VARCHAR(255)
                        );""")

        conn.commit()
        conn.close()
        print('database initalize complete')
    except Exception as e:
        print("while db_initialization, got ", e)

def sample_data_initialize():
    image_path = 'datasets/sample/images/'
    data_path = 'datasets/sample/sample_data.json'

    classes = ['Motorcycle_Pedestrian Road Violation', 'Motorcycle_No Helmet', 'Motorcycle_Jaywalking', 
    'Motorcycle_Signal Violation', 'Motorcycle_Stop Line Violation', 'Motorcycle_Crosswalk Violation', 
    'Bicycle_Pedestrian Road Violation', 'Bicycle_No Helmet', 'Bicycle_Jaywalking', 
    'Bicycle_Signal Violation', 'Bicycle_Stop Line Violation', 'Bicycle_Crosswalk Violation', 
    'Kickboard_Pedestrian Road Violation', 'Kickboard_No Helmet', 'Kickboard_Jaywalking', 
    'Kickboard_Signal Violation','Kickboard_Crosswalk Violation', 'Kickboard_Passenger Violation']

    try:
        with open(data_path,'r') as samplefile:
            sampledata = json.load(samplefile)
        conn = psycopg2.connect(**app.db_info)
        cursor = conn.cursor()

        for catid in sampledata:
            for list in sampledata[catid]:
                img_info = list[0]
                img_annotations = list[1]
                id = img_info['id']
                name = img_info['file_name']
                file_type = 'jpg'
                split = 'test'
                image = Image.open(image_path+name)
                image = image.tobytes()

                annotation_id = img_annotations['id']
                category_id = img_annotations['category_id']
                category_str = classes[category_id]
                bboxs = img_annotations['bbox']
                segmentation = []
                is_crowd = 0

                query1 = """
                        INSERT INTO aihub_img_table (id,name,type,split,image) 
                        VALUES (%s,%s,%s,%s,%s);
                        """
                query2 = """
                        INSERT INTO aihub_annotation_table (image_id,annotation_id,category_id,category_str,bboxs,segmentation,is_crowd) 
                        VALUES (%s,%s,%s,%s,%s,%s,%s);
                        """
                cursor.execute(query1,(id,name,file_type,split,psycopg2.Binary(image)))
                cursor.execute(query2,(id,annotation_id,category_id,category_str,bboxs,segmentation,is_crowd))


        conn.commit()

    # 데이터베이스 데이터 저장 확인
    # query = "SELECT * FROM aihub_img_table;"
    # cursor.execute(query)
    # result = cursor.fetchall()
    # for row in result:
    #     print(row)

    # query = "SELECT * FROM aihub_annotation_table;"
    # cursor.execute(query)
    # result = cursor.fetchall()
    # for row in result:
    #     print(row)

        conn.close()
        print('sample_data_loading successful')
    except Exception as e:
        print("while sample_data_loading, got ",e)

def sample_model_initialize():
    folder_path = 'datasets/sample/model/'
    modellist = os.listdir('datasets/sample/model')

    try:
        for filename in modellist:
            name = filename.split('.')[0]
            size_mb = round((os.path.getsize(folder_path+filename))/(1024*1024),3)
            format = filename.split('.')[1]
        
            conn = psycopg2.connect(**app.db_info)
            cursor = conn.cursor()
            query1 = """
                    INSERT INTO samplemodel_metadata_table (name,size_mb,format) 
                    VALUES (%s,%s,%s);
                    """
            cursor.execute(query1,(name,size_mb,format))
       
        conn.commit()

        # # 데이터베이스 데이터 저장 확인
        # query = "SELECT * FROM samplemodel_metadata_table;"
        # cursor.execute(query)
        # result = cursor.fetchall()
        # for row in result:
        #     print(row)

        conn.close()
        print('sample_model_loading successful')
    except Exception as e:
        print("while sample_model_loading, got ",e)

def app_initialize(db_info):
    print('app init start')
    # try:
    app.db_info = db_info
    # test_YOLO_model('yolov8')
    database_initialize(db_info)
    # sample_data_initialize()
    # sample_model_initialize()
    print('app_initialize successful')
    # except Exception as e:
        # print(e)

def uploading_model(uploaded_model):
    try:
        model_data = uploaded_model.read()  # 파일 내용을 읽어옴
        model_size = round(len(model_data)/(1024*1024),2)  # 파일 크기를 바이트로 읽고 메가바이트로 변환
        model_filename = uploaded_model.filename
        model_name = model_filename.split('.')[0]
        model_filetype = model_filename.split('.')[1]

        if model_filetype == 'pt':
            with open(f"datasets/uploaded/{uploaded_model.filename}", "wb") as f:
                f.write(model_data)
            f.close()
        else:
            raise('filetype:' + model_filetype +' is not supported')
    except Exception as e:
        raise(e)
    
    ####### saving model metadata to database
    try:
        conn = psycopg2.connect(**app.db_info)
        cursor = conn.cursor()
        query = """
                INSERT INTO model_metadata_table (name,size_mb,format) 
                VALUES (%s,%s,%s);
                """
        cursor.execute(query,(model_name,model_size,model_filetype))
        conn.commit()
        print('uploading model success')
    except Exception as e:
        conn.rollback()
        conn.close()
        raise(e)

def uploading_data(uploaded_data):
    support_imgtypes = ['png','jpg']
    support_vidtypes = ['mp4']
    try:     
        data_filename = uploaded_data.filename
        data = uploaded_data.read()  # 파일 내용을 읽어옴
        data_size = round(len(data)/(1024*1024),2)  # 파일 크기를 바이트로 읽고 메가바이트로 변환
        data_extension = data_filename.split('.')[1]
        data_format = None
        data_path = None
    except Exception as e:
        raise e
    
    if data_size > 200:
        raise ERRFILESIZE
    
    try:
        if data_extension in support_imgtypes:
            data_format = 'image'
            data_path = f"datasets/uploaded/images/{data_filename}"
            with open(f"datasets/uploaded/images/{data_filename}", "wb") as f:
                f.write(data)
            f.close()
        elif data_extension in support_vidtypes:
            data_format = 'video'
            data_path = f"datasets/uploaded/videos/{data_filename}"
            with open(f"datasets/uploaded/videos/{data_filename}", "wb") as f:
                f.write(data)
            f.close()
        else:
            raise ERRFILEFORMAT
    except Exception as e:
        if e == ERRFILEFORMAT:
            raise ERRFILEFORMAT
        else:
            try:
                if data_extension in support_imgtypes:
                    os.remove(f"datasets/uploaded/images/{data_filename}")
                else:
                    os.remove(f"datasets/uploaded/videos/{data_filename}")
            except:
                raise e
            finally:
                raise e

    ####### saving model metadata to database
    try:
        conn = psycopg2.connect(**app.db_info)
        cursor = conn.cursor()
        query = """
                INSERT INTO upload_data_table (file_name,size_mb,extension,format) 
                VALUES (%s,%s,%s);
                """
        cursor.execute(query,(data_filename,data_size,data_extension,data_format))
        conn.commit()
        print('uploading data success')
    except Exception as e:
        conn.rollback()
        conn.close()
        raise(e)
    finally:
        return data_filename,data_path,data_format

# def load_uploaded_model(model_name,file_format):
#     model = torch.load(f"datasets/uploaded/model/{model_name+'.'+ file_format}")
#     return model
    
# def load_sample_model(model_name,file_format):
#     model = torch.load(f"datasets/sample/model/{model_name+'.'+ file_format}")
#     return model

# def test_YOLO_model(model_name):
#     path = 'datasets/sample/model/' + model_name + '.pt'
#     model = YOLO(path)
#     result = model.val(data = "datasets/sample/yolo_train.yaml")
#     # return result

# ONNX모델과 이미지를 받아서 추론하는 함수
def Yolo_onnx_image_inference(model:cv2.dnn.Net, img, conf=0.25, nms_th=0.8):
    """
    ONNX 모델 받아서 추론

    returns:
        result : bbox표기된 이미지 (width, height, RGB)
        infer_time : 추론시간 (ms)
    """
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 128, 128),
          (128, 255, 128), (128, 128, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128),
          (192, 64, 0), (192, 192, 64), (64, 192, 192), (64, 64, 192), (192, 64, 192),
          (64, 192, 64), (255, 192, 128), (128, 255, 192), (128, 192, 255)]

    LABEL_NAMES = ['인도', '횡단보도', '자전거 도로', '교차로', '중앙 차선', '안전지대',
              '정지선', '정지선 위반 판별구역', '보행자 신호등 녹색', '보행자 신호등 적색',
              '차량 신호등 녹색', '차량 신호등 적색', '오토바이', '오토바이_보행자도로 통행위반',
              '오토바이_안전모 미착용', '오토바이_무단횡단', '오토바이_신호위반', '오토바이_정지선위반',
              '오토바이_횡단보도 주행위반', '자전거', '자전거 캐리어', '자전거_보행자도로 통행위반',
              '자전거_안전모 미착용', '자전거_무단횡단', '자전거_신호위반', '자전거_정지선위반',
              '자전거_횡단보도 주행위반', '킥보드', '킥보드 캐리어', '킥보드_보행자도로 통행위반',
              '킥보드_안전모 미착용', '킥보드_무단횡단', '킥보드_신호위반', '킥보드_횡단보도 주행위반',
              '킥보드_동승자 탑승위반']

    CLASS_NAMES = LABEL_NAMES[12:]
    [height, width, _] = img.shape
    length = max((height, width))
    resized_img = np.zeros((length, length, 3), np.uint8)
    resized_img[0:height, 0:width] = img
    scale = length / 640

    blob = cv2.dnn.blobFromImage(resized_img, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    # 출력 형태 (batch, 27, 8400)
    # 여기서 27은 차례대로 bbox좌표4개(x,y,w,h) + 클래스 23개
    # bbox의 숫자 8400개
    t1 = time.time()
    outputs = model.forward()
    t2 = time.time()
    infer_time = round((t2-t1)*1000, 2)

    # 출력형태 (8400, 27)로 변환 후 agnostic NMS 진행
    outputs = outputs[0].transpose()

    boxes = []
    cls_ids = []
    scores = []
    # 기준신뢰도 이상의 box만 추출
    for row in outputs:
        cls_score = row[4:]
        x, y, w, h = row[:4]
        # 최대 최소 및 최대 index 구하기
        minScore, maxScore, (_, minClassLoc), (_, maxClassIndex) = cv2.minMaxLoc(cls_score)
        if maxScore <= conf: # 기준 신뢰도 이하면 버리고 다음 row
            continue
        # left, top, width, height로 변환 (NMSBoxes input format)
        x1, y1= x-(0.5*w), y-(0.5*h)
        boxes.append([x1,y1,w,h])
        scores.append(maxScore)
        cls_ids.append(maxClassIndex)

    # agnostic NMS (선택된 bbox index list출력)
    nms_indices = cv2.dnn.NMSBoxes(boxes, scores, conf, nms_th)
 
    # 선택된 bbox 그리기 (cv2는 한글폰트 적용 x => PIL 사용)
    img = PIL.Image.fromarray(img[..., ::-1])
    font = PIL.ImageFont.truetype("static/assets/font/batang.ttc", 30)
    draw = PIL.ImageDraw.Draw(img, 'RGB')
    for idx in nms_indices:
        cls_name = CLASS_NAMES[cls_ids[idx]]
        score = scores[idx]
        x1, y1, w, h = np.array(boxes[idx]) * scale # 640*640에 원본스케일 곱하기
        color = COLORS[cls_ids[idx]]

        draw.rectangle((x1,y1,x1+w,y1+h), outline=color, width=5)
        tbbox = draw.textbbox([x1, y1-30], f'{score:.2f} '+cls_name, font=font)
        draw.rectangle(tbbox, fill=color)
        draw.text([x1, y1-30], f'{score:.2f} '+cls_name, font=font, fill='black')

    # 640*640*RGB 행렬및 추론시간(ms) 반환
    result = np.asarray(img)
    return result, infer_time

def image_to_base64(image_data):
    image_pil = Image.fromarray(image_data)
    image_bytes = io.BytesIO()
    image_pil.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    return base64.b64encode(image_bytes.read()).decode('utf-8')

def predict_video_with_onnx(video_path,model_name):
    model_path = 'datasets/sample/model/' + model_name + '.onnx'
    onnx_model = cv2.dnn.readNetFromONNX(model_path)
    save_base_dir = 'datasets/outputs/'

        # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 동영상 저장을 위한 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(save_base_dir + video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        # 프레임 읽기
        success, frame = cap.read()

        # 성공시 프레임 추론
        if success:
            results, infer_time = Yolo_onnx_image_inference(onnx_model, frame, conf=0.2)

            # cv2.namedWindow('YOLOv8m Tracking', cv2.WINDOW_NORMAL)
            # cv2.imshow("YOLOv8m Tracking", results[..., ::-1])  # RGB => BGR
            # print(f'추론시간 {infer_time}ms')

            # 결과 프레임을 동영상으로 저장
            output_video.write(results)

            # 키보드 q 누르면 중간 종료
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     cv2.destroyAllWindows()
            #     cap.release()
            #     output_video.release()
            #     break

        # 프레임 전부 읽어들이면 종료
        else:
            cv2.destroyAllWindows()
            cap.release()
            output_video.release()
            break
    print('predict successful')
    return save_base_dir + video_path, infer_time

def thread_predict_image_with_onnx(file_name,image_path,model_name):
    model_path = 'datasets/sample/model/' + model_name + '.onnx'
    onnx_model = cv2.dnn.readNetFromONNX(model_path)
    output_path = 'static/outputs/' + file_name
    test_img = cv2.imread(image_path)
    set_progress(10)
    result, infer_time = Yolo_onnx_image_inference(onnx_model, test_img, conf=0.5)
    cv2.imwrite(output_path,result)
    print('predict successful')
    set_progress(100)
    return result, infer_time

def thread_predict_video_with_onnx(file_name,video_path,model_name):
    model_path = 'datasets/sample/model/' + model_name + '.onnx'
    onnx_model = cv2.dnn.readNetFromONNX(model_path)
    save_base_dir = 'static/outputs/'

    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_time = total_frames / frame_rate


    # 동영상 저장을 위한 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(save_base_dir + file_name, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        # 프레임 읽기
        success, frame = cap.read()

        # 현재 진행률 계산
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        set_progress(int((current_frame / total_frames) * 100))

        # 성공시 프레임 추론
        if success:
            results, infer_time = Yolo_onnx_image_inference(onnx_model, frame, conf=0.2)

            # cv2.namedWindow('YOLOv8m Tracking', cv2.WINDOW_NORMAL)
            # cv2.imshow("YOLOv8m Tracking", results[..., ::-1])  # RGB => BGR
            # print(f'추론시간 {infer_time}ms')

            # 결과 프레임을 동영상으로 저장
            output_video.write(results)

            # 키보드 q 누르면 중간 종료
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     cv2.destroyAllWindows()
            #     cap.release()
            #     output_video.release()
            #     break

        # 프레임 전부 읽어들이면 종료
        else:
            cv2.destroyAllWindows()
            cap.release()
            output_video.release()
            break

    set_progress(100)
    print('predict successful')
    return save_base_dir + file_name, infer_time

def thread_predict_video_with_yolo_pt(file_name,video_path,model_name):
    model_path = 'datasets/sample/model/' + model_name + '.pt'
    model = YOLO(model_path)
    save_base_dir = 'static/outputs/'

    # 동영상 저장을 위한 설정
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(save_base_dir + file_name, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
    cap.release()

    progress = 10
    set_progress(progress)
    detection_results = model.predict(source=video_path,device = 'cpu')
    # img_num = len(detection_results)
    # count = 10
    # progress = int(count/(img_num+11))
    progress = 70
    set_progress(progress)
    for r in detection_results:
        # count+=1
        # progress = int(count/(img_num+11))
        # set_progress(progress)
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im_cv2 = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        output_video.write(im_cv2)
    output_video.release()

    set_progress(100)
    print('predict successful')
    return save_base_dir + file_name

# def generate_frames_yolo_pt(filename):
#     cap = cv2.VideoCapture(filename)
#     model = 
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Process frame
#         processed_frame = process_frame(frame)
        
#         # Encode frame as JPEG
#         _, buffer = cv2.imencode('.jpg', processed_frame)
#         frame = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
#     cap.release()

def generate_frames_yolo_onnx(filename):
    model_path = 'datasets/sample/model/' + 'YOLOv8m' + '.onnx'
    onnx_model = cv2.dnn.readNetFromONNX(model_path)
    cap = cv2.VideoCapture(filename)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # total_time = total_frames / frame_rate

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        set_progress(int((current_frame / total_frames) * 100))
        
        # Process frame
        results, infer_time = Yolo_onnx_image_inference(onnx_model, frame, conf=0.2)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', results)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    set_progress(100)

# main page
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['GET','POST'])
def model_simulate():
    if request.method == 'POST':
        uploaded_data = request.files['input_file']
        try:
            filename, filepath, file_format = uploading_data(uploaded_data)
        except Exception as e:
            return render_template('error_handle.html', message=str(e)) 
        return redirect(url_for('loading',filename = filename , filepath = filepath, file_format = file_format,init='0'))
    return render_template('simulate.html')

@app.route('/service')
def about_service():
    return render_template('service.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/taskdata')
def taskdata():
    return render_template('old/taskdata.html')


# @app.route('/result')
# def result():
#     file_format = request.args.get('file_format')
#     output_path = request.args.get('output_path')
#     response = make_response(send_from_directory('static', output_path))
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     if file_format == 'image':
#         # 이미지 결과 로직 및 템플릿 렌더링
#         return render_template('result_image.html',output_path = output_path)
#     elif file_format == 'video':
#         # 비디오 결과 로직 및 템플릿 렌더링
#         return render_template('result_video.html',output_path = output_path)
#     else:
#         return render_template('error_handle.html', message='Invalid file type')

@app.route('/result')
def result():
    file_format = request.args.get('file_format')
    output_path = request.args.get('output_path')
    response = make_response(send_from_directory('static', output_path))
    response.headers['Access-Control-Allow-Origin'] = '*'
    if file_format == 'image':
        # 이미지 결과 로직 및 템플릿 렌더링
        return render_template('result_image.html',output_path = output_path)
    elif file_format == 'video':
        # 비디오 결과 로직 및 템플릿 렌더링
        redis_client.set('output_path',output_path)
        return render_template('result_video2.html')
        # return render_template('result_video2.html',output_path = output_path)
    else:
        return render_template('error_handle.html', message='Invalid file type')    

@app.route('/loading',methods=['GET'])
def loading():
    init = request.args.get('init')
    if init == '0':
        reset_progress()
        filename = request.args.get('filename')
        filepath = request.args.get('filepath')
        file_format = request.args.get('file_format')
        
        try:
            if file_format == 'image':
                output_path = "outputs/"+filename
                inference_thread = threading.Thread(target = thread_predict_image_with_onnx,args = (filename,filepath, 'YOLOv8m',))
                inference_thread.start()
                return redirect(url_for('loading',output_path = output_path, file_format = file_format,init='1'))
            # else:
            #     output_path = "outputs/"+filename
            #     inference_thread = threading.Thread(target = thread_predict_video_with_onnx,args = (filename,filepath, 'YOLOv8m',))
            #     inference_thread.start()
            #     return redirect(url_for('loading',output_path = output_path, file_format = file_format,init='1'))
            else:
                output_path = "outputs/"+filename
                inference_thread = threading.Thread(target = thread_predict_video_with_yolo_pt,args = (filename,filepath, 'YOLOv8m',))
                inference_thread.start()
                return redirect(url_for('loading',output_path = output_path, file_format = file_format,init='1'))
            # else:
            #     redis_client.set('filename', filename)
            #     return render_template('result_video2.html')
            #     # return redirect(url_for('video',filename = filename))
            
        except Exception as e:
            return render_template('error_handle.html', message=str(e))  
    else:
        file_format = request.args.get('file_format')
        output_path = request.args.get('output_path')
        progress = int(get_progress())
  
        if progress >= 100:
            return redirect(url_for('result', output_path = output_path,file_format = file_format))       
        else:
            return render_template('loading.html', init='1', progress=progress,file_format = file_format,output_path = output_path)


@app.route('/video')
def video():
    file_path = 'static/'+ redis_client.get('output_path')
    print(file_path)
    return send_file(file_path, mimetype='video/mp4')



# @app.route('/video')
# def video():
#     filename = redis_client.get('filename')
#     return Response(generate_frames_yolo_onnx(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_progress')
def endpoint_get_progress():
    progress = get_progress()
    return jsonify({'progress': progress})

# @app.route('/data_upload', methods=['POST'])
# def data_upload():
#     uploaded_data = request.files['input_file']
#     try:
#         filename,filepath,file_format = uploading_data(uploaded_data)
#     except Exception as e:
#         if e == psycopg2.IntegrityError:
#             message = '중복된 데이터 이름이 데이터베이스에 존재합니다.'
#             return render_template('error_handle.html',message = message)
#         elif e == ERRFILESIZE:
#             message = ERRFILESIZE
#             return render_template('error_handle.html',message = message)
#         elif e == ERRFILEFORMAT:
#             message = ERRFILEFORMAT
#             return render_template('error_handle.html',message = message)
#         else:
#             message = 'while uploading model, got ' + str(e)
#             return render_template('error_handle.html',message = message)
        
#     try:
#         if file_format == 'image':
#             image_result, infer_time = predict_image_with_onnx(filename,'YOLOv8m')
#         else:
#             message = '미구현 '
#             return render_template('error_handle.html',message = message)
#     except Exception as e:
#         message = e
#         return render_template('error_handle.html',message = message)
#     # try:
#     #     os.remove(f"data/uploaded/{uploaded_model.filename}")
#     # except:
#     #     pass

#     ####### showing result on the page

#     ##############################################
#         return redirect(url_for('old/modeltesting'))


# @app.route('/modelupload', methods=['POST'])
# def model_upload():
#     uploaded_model = request.files['model']
#     try:
#         uploading_model(uploaded_model)
#     except psycopg2.IntegrityError as e:
#         print('중복된 모델명이 데이터베이스에 존재합니다.')
#     except Exception as e: 
#         print('while uploading model, got ',e)
#     finally:
#         try:
#             os.remove(f"data/uploaded/{uploaded_model.filename}")
#         except:
#             pass

#     ####### showing result on the page

#     ##############################################
#         return redirect(url_for('old/modeltesting'))
    

# main
if __name__ == '__main__':
    DB_postgresql = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
    }
    app_initialize(DB_postgresql)

    # options = {
    #     'workers': 4,          # 워커 프로세스 수
    #     'bind': '0.0.0.0:8000',# 바인딩 주소와 포트
    #     'timeout': 100,         # 워커 타임아웃 설정 (단위: 초)
    # }
    # FlaskGunicornApp().run(**options)
    # app.run(debug=True, host=DB_postgresql['host'])
    app.run(debug=True, host=DB_postgresql['host'])

