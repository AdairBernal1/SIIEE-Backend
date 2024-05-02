import datetime
from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from flask_mysqldb import MySQL
import threading
import cv2
from collections import Counter
import os
import mysql.connector
import json

import deepface.DeepFace as DeepFace



config = {
  'user': 'user1',
  'password': '123',
  'host': 'localhost',
  'database': 'siiee_bd',
  'raise_on_warnings': True,
}

cnx = mysql.connector.connect(**config)

app = Flask(__name__)
CORS(app)

is_recording = False

def getIDEstimulo(EstimuloName):
    cur = cnx.cursor()
    select_query = """SELECT ID FROM estimulos WHERE Estimulo = %s"""
    cur.execute(select_query, (EstimuloName,))
    IDEstimulo = cur.fetchone()
    cur.close()
    return IDEstimulo[0] if IDEstimulo else None


def getVideoPath(IDEval, IDEstimulo):
    cur = cnx.cursor()
    select_query = """SELECT VideoPath FROM Recordings WHERE IDEval = %s AND IDEstimulo = %s"""
    cur.execute(select_query, (IDEval, IDEstimulo))
    video_path = cur.fetchone()
    cur.close()
    return video_path[0] if video_path else None

def record_video(directory, filename, IDEval, IDEstimulo):
    global is_recording
    global video_file_path
    is_recording = True

    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename, ext = os.path.splitext(filename)
        file_path = os.path.join(directory, f"{filename}_{timestamp}{ext}")

    video_file_path = file_path
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(file_path, fourcc, 60.0, (640, 480))

    if not out.isOpened():
        print("Cannot open VideoWriter")
        return

    while(cap.isOpened() and is_recording):
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
        
    is_recording = False
    cv2.destroyAllWindows()
    cap.release()
    out.release()

    try:
        cur = cnx.cursor()
        insert_query = """INSERT INTO Recordings (IDEval, IDEstimulo, VideoPath)
                          VALUES (%s, %s, %s)"""
        cur.execute(insert_query, (IDEval, IDEstimulo, file_path))
        cnx.commit()
        cur.close()
    except Exception as e:
        print(f"Failed to insert video path into database: {e}")


@app.route('/start_recording', methods=['POST'])
def start_recording():
    directory = request.json.get('directory', 'C:/Users/adair/Desktop/SIIEE/backend/recordings')
    filename = request.json.get('filename', 'output.mp4') 
   
    IDEval = request.json.get('IDEval')
   
    EstimuloName = request.json.get('EstimuloName')
    IDEstimulo = getIDEstimulo(EstimuloName)
    
    if not is_recording:
        thread = threading.Thread(target=record_video, args=(directory, filename, IDEval, IDEstimulo))
        thread.start()
        return jsonify({"message": "Recording started"}), 200
    else:
        abort(400, description="Recording is already in progress")
        
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    if is_recording:
        is_recording = False
        return jsonify({"message": "Recording stopped", "file_path": video_file_path}), 200
    else:
        abort(400, description="No recording in progress")

def extract_frames(video_path, output_folder, analysis_frame_rate):
    cap = cv2.VideoCapture(video_path)
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_frame_rate / analysis_frame_rate)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_path = f"{output_folder}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()



def analyze_frame(frame_path):
    try:
        result = DeepFace.analyze(img_path=frame_path, actions=['emotion'], enforce_detection=False)
        emotion_data = result[0]
        emotion = emotion_data["dominant_emotion"]
        return emotion
    except ValueError:
        return "No face detected"

def process_video(frame_rate, video_path, output_folder):

    extract_frames(video_path, output_folder, 2)

    emotions = []
    frame_time = 0
    for frame_number, frame_filename in enumerate(os.listdir(output_folder)):
        frame_path = os.path.join(output_folder, frame_filename)
        emotion = analyze_frame(frame_path)
        if frame_number % frame_rate == 0:
            frame_time += 1
        emotions.append([frame_time, emotion])

    for frame_filename in os.listdir(output_folder):
        frame_path = os.path.join(output_folder, frame_filename)
        os.remove(frame_path)

    return emotions

@app.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    prueba_id = request.json.get('PruebaID')
    estudiante_id = request.json.get('EstudianteID')
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    cur = cnx.cursor()
    insert_query = """INSERT INTO log_evaluaciones (PruebaID, EstudianteID, createdAt)
                      VALUES (%s, %s, %s)"""
    cur.execute(insert_query, (prueba_id, estudiante_id, timestamp))
    cnx.commit()

    eval_id = cur.lastrowid
    cur.close()

    return jsonify({"message": "Evaluation started", "IDEval": eval_id}), 200


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    global video_file_path
    req_param = request.json
    output_folder = req_param.get('output_folder', "C:/Users/adair/Desktop/SIIEE/backend/recordings/frames" )
    frame_rate = req_param.get('frame_rate', 2) 
    IDEval = req_param.get('IDEval')
    EstimuloName = req_param.get('EstimuloName')
    IDEstimulo = getIDEstimulo(EstimuloName)
    video_path = getVideoPath(IDEval, IDEstimulo)

    emotions = process_video(frame_rate, video_file_path, output_folder)
    emotion_counter = Counter(emotion for _, emotion in emotions)
    predominant_emotion = emotion_counter.most_common(1)[0][0]

    # Call store_analysis
    store_analysis(IDEval, req_param.get('PruebaID'), IDEstimulo, predominant_emotion, emotions, req_param.get('RespuestaEstudiante'), video_path)

    return jsonify({"predominant_emotion": predominant_emotion, "full_analysis" : emotions}), 200



def store_analysis(eval_id, prueba_id, estimulo_id, emocion_predominante, resultados, respuesta_estudiante, video_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    resultados_str = json.dumps(resultados)

    cur = cnx.cursor()
    insert_query = """INSERT INTO analisis (IDEval, PruebaID, EstimuloID, EmocionPredominante, Resultados, RespuestaEstudiante, VideoPath, Timestamp)
                      VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
    cur.execute(insert_query, (eval_id, prueba_id, estimulo_id, emocion_predominante, resultados_str, respuesta_estudiante, video_path, timestamp))
    cnx.commit()

    cur.close()

    return {"message": "Analysis stored"}



@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error)}), 400

if __name__ == '__main__':
    app.run(debug=True)

# @app.route("/process")
# def process():
#     DeepFace.stream("C:/Users/adair/Desktop/SIIEE/backend/deepface/database", "Facenet512", "retinaface", time_threshold=2, frame_threshold=2)
#     return jsonify({"emotion": highest_emotion, "score": score})
    