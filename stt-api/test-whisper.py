from openai import OpenAI
import pprint
import time
client = OpenAI()
start_reading = time.time()
audio_file = open("stt-api/memento-sample-audio.mp3", "rb")
end_reading = time.time()
print(f"Reading audio file took {end_reading - start_reading} seconds")

start_transcription = time.time()
transcript = client.audio.transcriptions.create(
  file=audio_file,
  model="whisper-1",
  response_format="verbose_json",
  timestamp_granularities=["segment"]
)
end_transcription = time.time()
print(f"Transcription took {end_transcription - start_transcription} seconds")

print(transcript)
pprint.pp(str(transcript))