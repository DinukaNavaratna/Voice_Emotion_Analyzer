from flask import Flask, request
import sys
import subprocess
from flask_cors import CORS
import os
from loguru import logger
from datetime import datetime
from uuid import uuid4
from pydub import AudioSegment
from pydub.utils import make_chunks

from SER.Speech_Emotion_Recognition import initialize
from SER.Speech_Emotion_Recognition import analyze

app = Flask(__name__)
CORS(app)

model, lb = initialize()

@app.route("/analyze", methods=["POST", "GET"])
def check():
    f = request.files['file']
    ext = (f.filename).split('.')[-1]
    fileId = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
    filePath = "clips/"+fileId+"."+ext
    f.save(filePath)
    myaudio = AudioSegment.from_file("clips/"+fileId+"."+ext , "wav") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks as wav files
    filenames = []
    for i, chunk in enumerate(chunks):
        chunk_name = "clips/"+fileId+"_{0}.wav".format(i)
        #print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")
        filenames.append(chunk_name)

    #response = '{"file_path": "'+filePath+'", "file_type": "'+f.content_type+'", "emotion": "'+emotion+'"}'
    response = '{"results": ['
    first = True
    for filename in filenames:
        emotion = str(analyze(model, lb, filePath))
        if first:
            response += '{"file_path": "'+filename+'", "file_type": "audio/wav", "emotion": "'+emotion+'"}'
            first = False
        else:
            response += ', {"file_path": "'+filename+'", "file_type": "audio/wav", "emotion": "'+emotion+'"}'
        
    response += ']}'
    return response


@app.route("/train", methods=["POST", "GET"])
def train():
    from SER.Speech_Emotion_Recognition import initialize
    response = initialize()
    return response


@app.route("/", methods=["POST", "GET"])
def default():
    return "Voice Emotion Analyzer"

# run Server
if __name__ == "__main__":
    app.run(
		host="0.0.0.0",
		port=5000,
		debug=True,
        use_reloader=False
	)