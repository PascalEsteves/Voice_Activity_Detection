from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from typing import Optional
import requests
import os
import tempfile
from typing import ClassVar

app = FastAPI()

class Check_data(BaseModel):

    audio_url : str
    audio_id : int

class Pynoute:

    HYPER_PARAMETERS: ClassVar[dict] = {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    }

    def __init__(self, autentication:str) -> None:
        self.model = Model.from_pretrained("pyannote/segmentation-3.0",use_auth_token=autentication)
        self.pipeline = VoiceActivityDetection(segmentation=self.model).instantiate(Pynoute.HYPER_PARAMETERS)

    def segment_audio(self, audio_link:str, audio_id:input):

        response = requests.get(audio_link)
        result = {}

        if response.status_code==200:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(response.content)
                temp_audio_file_path = temp_audio_file.name

            vad = self.pipeline(temp_audio_file_path)

            segments=[]

            for index,segment in enumerate(vad.itersegments()):

                segments.append({"Audio_id": int(audio_id),
                                 "Start_time": float(segment.start),
                                 "End_time": float(segment.end),
                                 "Segment_number": int(index+1)
                                 })
            os.remove(temp_audio_file_path)

            return segments
        else:
            return {"error": "Error downloading audio file"}


async def get_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Invalid Authorization header format")

    api_key: str = authorization[len("Bearer "):]
    return api_key

@app.get("/audio_segmentation")
async def audio_segmentation(check_data:Check_data, api_key: str = Depends(get_api_key)):

    speach_det = Pynoute(autentication=api_key)
    result = speach_det.segment_audio(audio_link=check_data.audio_url, audio_id=check_data.audio_id)

    return result


