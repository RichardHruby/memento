import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import Optional, List, Dict, Any, Union


MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://chaitanyaparsana:fLkvD1VHzvTTt9N2@memento-hack-cluster.kgqm9me.mongodb.net/?retryWrites=true&w=majority&appName=Memento-hack-cluster"

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "memento-db"
USER_COLLECTION_NAME = "users"
CONVO_COLLECTION_NAME = "conversations"

USER_MONGODB_COLLECTION = client[DB_NAME][USER_COLLECTION_NAME]
CONVO_MONGODB_COLLECTION = client[DB_NAME][CONVO_COLLECTION_NAME]

def get_conversations(user_name: str) -> Union[List[Dict[str, Any]], str]:
    """Get conversations for a specific user.

    Args:
        user_name (str): The name of the user.

    Returns:
        Union[List[Dict[str, Any]], str]: A list of conversations if the user is found, otherwise a string indicating the user was not found.
    """
    # Fetch user
    user = USER_MONGODB_COLLECTION.find_one({"name": user_name})
    if user:
        # Fetch conversations
        user_id = user['_id']
        conversations = CONVO_MONGODB_COLLECTION.find({"persons": ObjectId(user_id)})
        
        # Return conversations
        return list(conversations)
    else:
        return "User not found"
    

st.title("""Retrieving Conversations from a user's 'Memory'""")
st.write("This is a simple example of how to retrieve a user's conversations from a database.")

user_name = st.text_input("Enter your name", "John Doe")

if st.button("Retrieve Conversations"):
    conversations = get_conversations(user_name)
    if isinstance(conversations, str):
        st.write(conversations)
    else:
        st.write([c.get("summary") for c in conversations])