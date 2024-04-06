
import streamlit as st
import time
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from openai import OpenAI


from typing import List, Optional, Dict, Any

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
from bson.objectid import ObjectId

from streamlit_mic_recorder import mic_recorder

import io


MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_URI")

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "memento-db"
USER_COLLECTION_NAME = "users"
CONVO_COLLECTION_NAME = "conversations"

USER_MONGODB_COLLECTION = client[DB_NAME][USER_COLLECTION_NAME]
CONVO_MONGODB_COLLECTION = client[DB_NAME][CONVO_COLLECTION_NAME]


def upload_conversation(conversation_dict: Dict[str, Any]) -> None:
    """Uploads a conversation to the MongoDB database.

    Args:
        conversation_dict: A dictionary containing the conversation data.
    """
    # Iterate over persons
    for person in conversation_dict['persons']:
        # Check if person exists
        existing_person = USER_MONGODB_COLLECTION.find_one({"name": person['name']})
        if existing_person:
            # Get person_id
            person_id = existing_person['_id']
            
            # Update facts
            new_facts = [fact for fact in person['facts'] if fact not in existing_person['facts']]
            if new_facts:
                USER_MONGODB_COLLECTION.update_one({"_id": person_id}, {"$push": {"facts": {"$each": new_facts}}})
        else:
            # Insert person and get _id
            person_id = USER_MONGODB_COLLECTION.insert_one(person).inserted_id

        # Prepare conversation data
        conversation = {
            "persons": [ObjectId(person_id)],
            "timestamp": conversation_dict['timestamp'],
            "summary": conversation_dict['summary']
        }

        # Check if conversation exists
        existing_conversation = CONVO_MONGODB_COLLECTION.find_one({"persons": conversation['persons'], "timestamp": conversation['timestamp']})
        if existing_conversation:
            # Update conversation
            CONVO_MONGODB_COLLECTION.update_one({"_id": existing_conversation['_id']}, {"$set": {"summary": conversation['summary']}})
        else:
            # Insert conversation
            CONVO_MONGODB_COLLECTION.insert_one(conversation)


chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")


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
    # Set up OpenAI API client
    client = OpenAI()
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


st.title('Capture social interactions :)')

# Audio recording
audio = record_audio()

if audio is not None:
    # Transcribe audio using OpenAI API
    st.session_state.transcript = transcribe_audio(audio)


if st.button("START PROCESSING!"):
    
    
    with st.status("Processing...", expanded=True) as status:
        
        st.write("Running speech to text...")
        
        if st.session_state.transcript is not None:
            # Display the formatted text
            formatted_text = format_transcript(st.session_state.transcript)

            st.write(formatted_text)
    
    
        st.write("Detecting conversation...")
        time.sleep(1)


        class Person(BaseModel):
            """Information about a person."""

            name: str = Field(..., description="The name of the person. This field is required.")
            image_url: Optional[str] = Field('', description="URL of an image of the person. This is usually left empty.")
            facts: List[str] = Field(default_factory=list, description="A list of miscellaneous facts about the person.")

        class Conversation(BaseModel):
            """Model for capturing conversation details.

            This class captures the essentials of a conversation, including who was involved, when it occurred, and a summary of the discussion. The timestamp is optional and can be left empty, aligning with the requirement to occasionally omit the specific time of the conversation.
            """

            persons: List[Person]
            timestamp: Optional[str] = Field(None, description="The timestamp when the conversation occurred. This field is always empty.")
            summary:Optional[str] = Field(..., description="A bulleted summary list of key points discussed in the conversation.")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert extraction algorithm. "
                    "Only extract relevant information from the text. "
                    "If you do not know the value of an attribute asked to extract, "
                    "return null for the attribute's value."
                    "before you formulate your response think about who the individual speakers are and enclose your thoughts in thinking tags"
                ),
                # Please see the how-to about improving performance with
                # reference examples.
                # MessagesPlaceholder('examples'),
                ("human", "{text}"),
            ]
        )

        runnable = prompt | chat.with_structured_output(schema=Conversation)

        st.write("Running extraction...")
        convo_obj = runnable.invoke({"text": formatted_text})

        st.write("Syncing data to MongoDB...")

        upload_conversation(convo_obj.dict())

        status.update(label="Download complete!", state="complete", expanded=False)


