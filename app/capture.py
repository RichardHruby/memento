
import streamlit as st
import time
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from typing import List, Optional, Dict, Any

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
from bson.objectid import ObjectId


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

st.title('Capture social interactions :)')

if st.button("START PROCESSING!"):
    
    
    with st.status("Processing...", expanded=True) as status:
        
        st.write("Running speech to text...")
        
        time.sleep(1)
        

        transcript = """
            "Hey man, how\'s it going? Good, good. How about you? '
            "What's your favorite animal, man? Pretty good. My favorite animal is "
            "dinosaur. But let me start, like, what's your name? I'm Josh. What's yours? "
            "I'm Peter, by the way. Nice to meet you. So how do you like the conference? "
            "It's pretty cool. It's really big. I'm meeting a lot of really cool people. "
            "My favorite animal is the capybara, so I've been finding a lot of capybara "
            "lovers around here. It's been nice. Tell me more about capybara. They're "
            'just really goofy. Interesting. Is that why you like them? Yeah, yeah. '
            "It's... That's very interesting, man. Like, I've never seen someone who's a "
            "fan of capybara. It's the first time. Yeah, they're like huge rats, "
            "basically. Oh, okay. Interesting. They're really cute, and there's been a "
            'trend going on, going around on, like, Instagram and stuff of reels of '
            "capybaras doing random stuff. They're like pandas, by the way. That's "
            'amazing. It kind of reminds me of how, you know, people use animal names '
            "for, like, technology stuff. Like, you know, I don't want to quite say LLM, "
            "but, you know, llama is something. Yeah. So, but that's very interesting. "
            "I'm a huge fan of dinosaur, and one of the popular things is just, I think "
            "they're very cool. I used to be a very much a fan of a velociraptor, which "
            'is like, you know, a dinosaur, right? Yeah. What really sparked your '
            "interest in velociraptors? Was it Jurassic Park? I could, it's partly the "
            "Jurassic Park, but I would say it's more like the way they behave. Something "
            "that was very attractive is just like, you know, they're very intelligent. "
            "They work as a group, and, you know, they're kind of excited, like, it's "
            "kind of like they're not very big, but they're like super genius, right? "
            'Yeah. Like the dog, yes, but, all right. I hope you have a good conference, '
            'man. Yeah, you too, man. I\'ll see you around. See ya."
        """

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
        convo_obj = runnable.invoke({"text": transcript})

        st.write("Syncing data to MongoDB...")

        upload_conversation(convo_obj.dict())

        status.update(label="Download complete!", state="complete", expanded=False)


