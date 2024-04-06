import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import pprint
import time
import ast
import re
import io
import os

# Set up OpenAI API client
client = OpenAI()

# Streamlit app
def main():
    st.title("Audio Transcription App")

    # Audio recording
    audio = record_audio()

    if audio is not None:
        # Transcribe audio using OpenAI API
        transcript = transcribe_audio(audio)

        if transcript is not None:
            # Display the formatted text
            formatted_text = format_transcript(transcript)
            st.subheader("Transcription Result")
            st.text(formatted_text)

def record_audio():
    st.subheader("Audio Recording")
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        format="webm",
        key="audio_recording"
    )
    return audio

def transcribe_audio(audio):
    try:
        start_transcription = time.time()
        audio_bio = io.BytesIO(audio['bytes'])
        audio_bio.name = 'audio.webm'
        transcript = client.audio.transcriptions.create(
            file=audio_bio,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        end_transcription = time.time()
        st.info(f"Transcription took {end_transcription - start_transcription} seconds")
        return transcript
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def format_transcript(transcript):
    try:
        speaker = 0
        cleaned_text = []
        for segment in transcript.segments:
            if speaker == 0:
                cleaned_text.append(f"Speaker A: {segment['text']}")
                speaker = 1
            else:
                cleaned_text.append(f"Speaker B: {segment['text']}")
                speaker = 0
        formatted_text = " \n\n".join(cleaned_text)
        return formatted_text
    except Exception as e:
        st.error(f"Error during transcript formatting: {str(e)}")
        return ""

if __name__ == "__main__":
    main()