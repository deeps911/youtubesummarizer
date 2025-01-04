# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:37:45 2025

@author: dverma
"""

import os
from config import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

#import openai
from dotenv import load_dotenv # Add
load_dotenv() # Add
import langchain_openai
from langchain_openai import OpenAI
import streamlit as st

import re
from youtube_transcript_api import YouTubeTranscriptApi




CHUNK_SIZE = 2000  # Approx. characters per chunk
OVERLAP = 200      # Approx. character overlap between chunks

# Create an OpenAI LLM instance
llm = OpenAI(
    max_tokens=200,                # Let's keep the output short per chunk
    temperature=0.7
)


# 3. Function to Chunk Transcript Text
# -------------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Splits a large text into overlapping chunks of length `chunk_size`.
    Overlap ensures we don’t lose context at the chunk boundaries.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        # Advance start, ensuring overlap
        start += (chunk_size - overlap)
    
    return chunks




# 4. Retrieve YouTube Transcript
# -------------------------------------------------------------
def get_youtube_transcript(video_url: str) -> str:
    """
    Extracts the transcript from a YouTube video URL and returns it as a string.
    """
    # Extract video ID from URL
    video_id_match = re.search(r"v=([^&]*)", video_url)
    if not video_id_match:
        raise ValueError("Invalid YouTube URL, couldn't extract video ID.")
    
    video_id = video_id_match.group(1)
    
    # Retrieve the transcript (list of dicts)
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
    
    # Convert to a single string
    transcript_text = " ".join([item["text"] for item in transcript_data])
    return transcript_text




def summarize_transcript(transcript: str) -> str:
    """
    Breaks the transcript into chunks, summarizes each chunk,
    and then combines the partial summaries into a final summary.
    """
    # 1. Split into chunks
    chunks = chunk_text(transcript, CHUNK_SIZE, OVERLAP)

    # 2. Summarize each chunk
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        prompt_chunk = (
            "You are a helpful assistant. Summarize in bullet form the following text:\n\n"
            f"{chunk}\n\n"
        )
        #print(prompt_chunk)
        summary = llm(prompt_chunk)
        #print(summary)
        partial_summaries.append(summary.strip())
     
        
    # # 3. Combine partial summaries into a final summary
    final_prompt = (
        "Combine the following partial summaries into one cohesive, bullet-point summary: \n\n"
        f"{partial_summaries}\n\n"
    )
    final_summary = llm(final_prompt)
    return final_summary
    
    
        
        

     


# ------------------------------------
# 4. Streamlit App
# ------------------------------------
def main():
    st.title("YouTube Transcript Summarizer")
    st.write(
        "Enter a YouTube video URL below. This app will retrieve the transcript, "
        "chunk it, summarize each chunk, and then combine them into a final summary."
    )

    youtube_url = st.text_input("YouTube URL", "")
    if st.button("Summarize"):
        if youtube_url.strip():
            with st.spinner("Fetching transcript and summarizing..."):
                try:
                    transcript_text = get_youtube_transcript(youtube_url)
                    if not transcript_text:
                        st.error("No transcript found for this video.")
                    else:
                        summary_result = summarize_transcript(transcript_text)
                        st.subheader("Summary")
                        st.write(summary_result)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()   




#Main Code

# youtube_url = "https://www.youtube.com/watch?v=m_wZIMFKVO8"
# # Get the full transcript
# transcript_text = get_youtube_transcript(youtube_url)
# # Summarize it
# final_summary = summarize_transcript(transcript_text)    
# # Print the final summary
# print("\n=== FINAL SUMMARY ===")
# print(final_summary)   
