import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

st.title("Tamil Nadu Assembly - YouTube Political Analysis")

# -----------------------------
# LOAD DATA
# -----------------------------
comments_df = pd.read_csv("comments_cleaned.csv")
summary_df = pd.read_csv("sentiment_summary.csv", index_col=0)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("Sample Comments")
st.dataframe(comments_df.head(20))

# -----------------------------
# 1. Sentiment by Party
# -----------------------------
st.subheader("Sentiment Distribution by Party")

fig1, ax1 = plt.subplots()
summary_df.plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# -----------------------------
# 2. Overall Sentiment
# -----------------------------
st.subheader("Overall Sentiment")

overall = comments_df['sentiment'].value_counts()

fig2, ax2 = plt.subplots()
overall.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_ylabel("")
st.pyplot(fig2)

# -----------------------------
# 3. Party Mentions
# -----------------------------
st.subheader("Party Mentions")

all_entities = []
for ents in comments_df['entities']:
    if isinstance(ents, str):
        all_entities.extend(ents.split(","))

entity_counts = Counter(all_entities)

fig3, ax3 = plt.subplots()
ax3.bar(entity_counts.keys(), entity_counts.values())
st.pyplot(fig3)

# -----------------------------
# 4. Leader Mentions
# -----------------------------
st.subheader("Leader Mentions")

leaders = {
    "Stalin": ["stalin"],
    "EPS": ["eps", "palaniswami"],
    "Annamalai": ["annamalai"],
    "Vijay": ["vijay", "thalapathy"]
}

def detect_leaders(text):
    found = []
    for leader, keywords in leaders.items():
        for word in keywords:
            if word in text:
                found.append(leader)
                break
    return found

leader_list = []
for text in comments_df['cleaned']:
    leader_list.extend(detect_leaders(str(text)))

leader_counts = Counter(leader_list)

fig4, ax4 = plt.subplots()
ax4.bar(leader_counts.keys(), leader_counts.values())
st.pyplot(fig4)

# -----------------------------
# 5. Topics (Keywords)
# -----------------------------
st.subheader("Top Discussion Topics")

from sklearn.feature_extraction.text import CountVectorizer

#  clean NaN + empty values BEFORE vectorizer
clean_texts = comments_df['cleaned'].fillna("")
clean_texts = clean_texts[clean_texts.str.strip() != ""]

vectorizer = CountVectorizer(
    stop_words='english',
    max_features=10
)

X = vectorizer.fit_transform(clean_texts)

words = vectorizer.get_feature_names_out()
counts = X.toarray().sum(axis=0)

fig5, ax5 = plt.subplots()
ax5.bar(words, counts)
plt.xticks(rotation=45)
st.pyplot(fig5)