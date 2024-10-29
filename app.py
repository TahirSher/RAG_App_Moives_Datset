import os
import pandas as pd
import gzip
import requests
import io
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load the proprietary dataset from GitHub
def load_dataset_from_github(gzip_url):
    # Download the Gzip file
    response = requests.get(gzip_url)
    if response.status_code == 200:
        # Load the Gzip file and read the CSV
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            return pd.read_csv(f)
    else:
        raise Exception("Failed to download the dataset.")

# URL of the Gzip file containing the dataset on GitHub
gzip_url = "https://github.com/TahirSher/RAG_App_Moives_Datset/raw/main/compressed_data.csv.gz"
movies_df = load_dataset_from_github(gzip_url)

# Preprocess the dataset by creating summaries and vectors
def preprocess_data(df):
    df['summary'] = df.apply(lambda row: f"{row['title']} ({row['release_date']}): {row['overview']} "
                                         f"Genres: {row['genres']} Keywords: {row['keywords']}", axis=1)
    return df

movies_df = preprocess_data(movies_df)

# Convert summaries to TF-IDF vectors for retrieval
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies_df['summary'])

# Define function to retrieve similar movies based on a query
def retrieve_similar_movies(query, df, tfidf_matrix, top_n=5):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Call Groq API for generation based on the retrieved summaries and query
def generate_summary_with_groq(query, retrieved_text):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"{query}\n\nRelated information:\n{retrieved_text}"}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to handle different types of queries
def handle_query(query):
    # Check for specific types of queries
    if "details" in query.lower():
        # Return details about the movie(s)
        movie_title = query.split("details about")[-1].strip()
        details = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
        if not details.empty:
            return details.to_string(index=False)
        else:
            return "No details found for the specified movie."
    elif "list" in query.lower() and "movies" in query.lower():
        return movies_df['title'].tolist()[:10]  # Return first 10 movie titles as a simple list
    else:
        # Default to generating a summary for movie-related queries
        retrieved_movies = retrieve_similar_movies(query, movies_df, tfidf_matrix)
        retrieved_summaries = " ".join(retrieved_movies['summary'].values)
        return generate_summary_with_groq(query, retrieved_summaries)

# Streamlit Application
def main():
    st.title("Movie RAG-based Application")
    
    # Initialize session state variables
    if 'questions' not in st.session_state:
        st.session_state.questions = []
        
    if 'responses' not in st.session_state:
        st.session_state.responses = []

    # User input
    user_query = st.text_input("Ask a question about movies:")
    
    if user_query:
        # Check if user wants to exit
        if user_query.lower() in ['exit', 'no', 'quit']:
            st.write("Exiting the application. Goodbye!")
            return
        
        # Handle the user's query
        generated_response = handle_query(user_query)
        
        # Store the question and response in session state
        st.session_state.questions.append(user_query)
        st.session_state.responses.append(generated_response)

        # Display the generated response
        st.subheader("Response:")
        st.write(generated_response)

        # Provide an option for the user to ask another question
        st.text("You can ask another question.")

        # Display the previous questions and responses
        if st.session_state.questions:
            st.write("### Previous Questions and Responses:")
            for question, response in zip(st.session_state.questions, st.session_state.responses):
                st.write(f"- **Q:** {question}")
                st.write(f"  **A:** {response}")

if __name__ == "__main__":
    main()
