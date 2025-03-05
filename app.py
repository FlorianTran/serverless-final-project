from flask import Flask, jsonify, request, send_file
import os
import io
from datetime import datetime
import boto3
import uuid
from gtts import gTTS
from pydub import AudioSegment
from inferrvc import RVC, load_torchaudio
import soundfile as sf
# Allow safe unpickling of the fairseq Dictionary class
import torch
from fairseq.data.dictionary import Dictionary

torch.serialization.add_safe_globals([Dictionary])

app = Flask(__name__)
	
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
S3_BUCKET_NAME = "voice-generator-bucket"
dbclient = boto3.client('dynamodb', endpoint_url="http://localhost:8000", region_name="us-east-1", aws_secret_access_key="fake", aws_access_key_id="fake")
s3client = boto3.client('s3', endpoint_url="http://localhost:4569", aws_secret_access_key="S3RVER", aws_access_key_id="S3RVER")

@app.route("/")
def hello():
    return """
<form action="/voice" method="POST">
    <input type="text" name="sentence" required>
    <button type="submit">Sumbit</button>
</form>
"""

@app.route("/voice", methods=["POST"])
def voice_post():
    
    sentence = request.form.get("sentence")

    # AI SETUP
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["RVC_MODELDIR"] = "DarthVaderCustom"
    os.environ["RVC_INDEXDIR"] = "DarthVaderCustom"
    os.environ["RVC_OUTPUTFREQ"] = "45000"
    os.environ["RVC_RETURNBLOCKING"] = "True"
    tts = gTTS(text=sentence)
    tts.save("temp.mp3")
    audio = AudioSegment.from_mp3("temp.mp3")
    audio.export("temp.wav", format="wav")

    model_path = "DarthVaderCustom_e1000_s45000.pth"
    index_path = "added_IVF510_Flat_nprobe_1_DarthVaderCustom_v2.index"
    darth_vader = RVC(model_path, index=index_path)

    aud, sr = load_torchaudio(temp_wav)

    converted_audio = darth_vader(
        aud, 5, output_device="cpu", output_volume=RVC.MATCH_ORIGINAL, index_rate=0.9
    )

    random_filename = f"{uuid.uuid4()}.mp3"

    buffer = io.BytesIO()
    torch.save(converted_audio, buffer)

    s3client.upload_fileobj(buffer, S3_BUCKET_NAME, random_filename)
    
    resp = dbclient.put_item(
        TableName = DYNAMODB_TABLE,
        Item = {
            'key' : { "S": random_filename},
            'createdAt' : {"S": datetime.now().isoformat()},
            'wave' : {"S": random_filename},
            'sentence' : {"S": sentence}
        }
    )

    return jsonify({"filename":random_filename}), 200

@app.route("/voice/<string:filename>", methods=["GET"])
def voice_get(filename):
    try:
        file_obj = s3client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        fc = file_obj['Body'].read()
        return send_file(io.BytesIO(fc), as_attachment=True, download_name=filename), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/voices")
def voices():
    return "Hello World!"