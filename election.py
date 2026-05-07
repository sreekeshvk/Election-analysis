import pandas as pd
import re
import emoji
import os
import torch
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline
from collections import defaultdict
from dotenv import load_dotenv

# -----------------------------
# CONFIG & API SETUP
# -----------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
QUERY = "Tamil Nadu Assembly"
MAX_VIDEOS = 100
COMMENTS_PER_VIDEO = 500

youtube = build('youtube', 'v3', developerKey=API_KEY)

# -----------------------------
# STEP 1: FAST MULTILINGUAL SENTIMENT MODEL
# -----------------------------
# Using XLM-RoBERTa (Cross-lingual) - Excellent for Tamil/English mix
print("Loading Multilingual Sentiment Model...")
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_pipe = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=-1) # set device=0 if you have GPU

def get_multilingual_sentiment(text):
    if not text or len(text) < 2:
        return "Neutral"
    try:
        # Truncate to 512 tokens to avoid model errors
        result = sentiment_pipe(text[:512])[0]
        label = result['label'].lower()
        if 'pos' in label: return "Positive"
        if 'neg' in label: return "Negative"
        return "Neutral"
    except:
        return "Neutral"

# -----------------------------
# STEP 2: VIDEO & COMMENT FETCHERS
# -----------------------------
def get_videos():
    videos = []
    next_page_token = None
    while len(videos) < MAX_VIDEOS:
        request = youtube.search().list(q=QUERY, part="snippet", type="video", maxResults=50, order="relevance", pageToken=next_page_token)
        response = request.execute()
        video_ids = [item['id']['videoId'] for item in response['items']]
        stats_response = youtube.videos().list(part="snippet,statistics", id=",".join(video_ids)).execute()
        for item in stats_response['items']:
            videos.append({
                "video_id": item['id'],
                "title": item['snippet']['title'],
                "description": item['snippet']['description'],
                "channel": item['snippet']['channelTitle'],
                "publish_date": item['snippet']['publishedAt'],
                "view_count": item['statistics'].get('viewCount', 0),
                "like_count": item['statistics'].get('likeCount', 0),
                "comment_count": item['statistics'].get('commentCount', 0)
            })
        next_page_token = response.get("nextPageToken")
        if not next_page_token: break
    return pd.DataFrame(videos[:MAX_VIDEOS])

def get_comments(video_id):
    comments = []
    next_page_token = None
    while len(comments) < COMMENTS_PER_VIDEO:
        try:
            request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=next_page_token, textFormat="plainText")
            response = request.execute()
            for item in response['items']:
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            next_page_token = response.get("nextPageToken")
            if not next_page_token: break
        except HttpError as e:
            if "commentsDisabled" in str(e): return []
            break
    return comments[:COMMENTS_PER_VIDEO]

# -----------------------------
# STEP 3: CLEANING & ENTITIES
# -----------------------------
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    # Keep Tamil (\u0B80-\u0BFF) and English/Numbers
    text = re.sub(r"[^\w\s\u0B80-\u0BFF]", "", text)
    return text.lower().strip()

entities = {
    "DMK": ["dmk", "stalin", "udhay", "ஸ்டாலின்"],
    "AIADMK": ["aiadmk", "eps", "palaniswami", "எடப்பாடி"],
    "BJP": ["bjp", "annamalai", "modi", "அண்ணாமலை"],
    "TVK": ["tvk", "vijay", "thalapathy", "விஜய்"]
}

def is_valid_comment(text):
    noise_words = {"fake", "ai", "ok", "lol", "wow", "poda", "dei"}
    words = text.split()
    if len(words) < 2 or text in noise_words: return False
    return True

def detect_entities(text):
    found = []
    for party, keywords in entities.items():
        if any(word in text for word in keywords):
            found.append(party)
    return found

# -----------------------------
# STEP 4: MAIN PIPELINE
# -----------------------------
print("Starting Data Extraction...")
videos_df = get_videos()
videos_df.to_csv("videos_master.csv", index=False, encoding="utf-8-sig")

all_comments = []
valid_video_count = 0
required_videos = 10

for _, row in videos_df.iterrows():
    if valid_video_count >= required_videos: break
    if "live" in row['title'].lower(): continue

    vid = row['video_id']
    raw_comments = get_comments(vid)
    
    if len(raw_comments) < 100: continue
    print(f"Processing Video: {vid} ({len(raw_comments)} comments)")

    for c in raw_comments:
        cleaned = clean_text(c)
        if not cleaned or not is_valid_comment(cleaned): continue

        sentiment = get_multilingual_sentiment(cleaned)
        ents = detect_entities(cleaned)

        all_comments.append({
            "video_id": vid,
            "original": c,
            "cleaned": cleaned,
            "sentiment": sentiment,
            "entities": ",".join(ents)
        })
    valid_video_count += 1

# -----------------------------
# STEP 5: SAVE & SUMMARY
# -----------------------------
if all_comments:
    comments_df = pd.DataFrame(all_comments)
    # Crucial: Remove NaNs before Streamlit/Analysis
    comments_df = comments_df.dropna(subset=['cleaned'])
    comments_df.to_csv("comments_cleaned.csv", index=False, encoding="utf-8-sig")

    summary = defaultdict(lambda: {"Positive":0, "Negative":0, "Neutral":0})
    for _, row in comments_df.iterrows():
        for ent in row['entities'].split(","):
            if ent: summary[ent][row['sentiment']] += 1

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv("sentiment_summary.csv")
    print("\n--- FINAL ANALYSIS ---")
    print(summary_df)
else:
    print("Zero comments processed. Check API Key or Filters.")