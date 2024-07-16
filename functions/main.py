import firebase_admin
from firebase_admin import firestore, storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import datetime

from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import json


# Initialize Firebase Admin
firebase_admin.initialize_app()

def tokenizer(x):
    return x

def train_poll_hashtags_kmeans_model(request):
    # Fetch poll data from Firestore
    db = firestore.client()
    polls_ref = db.collection('polls')
    all_polls_data = {doc.id: doc.to_dict() for doc in polls_ref.stream()}

    # Extract hashtags
    poll_hashtags = {key: value.get('hashtags', []) for key, value in all_polls_data.items()}
    all_polls_hashtags_list = [{'id': key, 'hashtags': value} for key, value in poll_hashtags.items()]

    # Vectorize hashtags
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
    X = vectorizer.fit_transform([post["hashtags"] for post in all_polls_hashtags_list])

    # Perform KMeans clustering
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)

    # Save the model and vectorizer to Firebase Storage
    model_data = {
        'kmeans': kmeans,
        'vectorizer': vectorizer,
        'data': all_polls_hashtags_list
    }

    bucket_name = 'askoverse-ml-models-bucket'  # Replace with your bucket name
    bucket = storage.bucket(bucket_name)

    # Create a temporary file to store the model data
    with tempfile.NamedTemporaryFile() as temp_model_file:
        
        pickle.dump(model_data, temp_model_file)
        
        temp_model_file.flush()

        # Save the current latest model to the backup folder with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_blob = bucket.blob(f'backup_models/kmeans_model_{timestamp}.pkl')
        backup_blob.upload_from_filename(temp_model_file.name)

        # Save the model as the latest model
        latest_blob = bucket.blob('models/latest_model.pkl')
        latest_blob.upload_from_filename(temp_model_file.name)
    


    return 'Model trained and saved successfully.'




def load_model_from_storage(bucket_name, model_path):
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(model_path)
    
    with tempfile.NamedTemporaryFile() as temp_model_file:
        blob.download_to_filename(temp_model_file.name)
        temp_model_file.seek(0)
        model_data = pickle.load(temp_model_file)
    
    return model_data

def get_top_n_user_hashtags(user_id, n):
    db = firestore.client()
    user_hashtags_collection_name = "user_hashtags"
    doc = db.collection(user_hashtags_collection_name).document(user_id).get()

    if not doc.exists:
        return []

    return sorted(doc.to_dict().items(), key=lambda item: item[1], reverse=True)[:n]


def recommend_user_interest_polls(request):
    data = request.get_json()
    user_id = data.get('user_id')
    no_of_polls = data.get('no_of_polls')

   
    db = firestore.client()
    model_path = 'models/latest_model.pkl'
    
    # Load the model, vectorizer, and poll_hashtags from Firebase Storage
    bucket_name = 'askoverse-ml-models-bucket'
    model_data = load_model_from_storage(bucket_name, model_path)

    
    
    kmeans = model_data['kmeans']
    vectorizer = model_data['vectorizer']
    all_polls_hashtags_list = model_data['data']

   
    # Get user preferences
    user_interest = get_top_n_user_hashtags(user_id, 50)
    user_preferences = [item[0] for item in user_interest]

   

    if not user_preferences:
        return json.dumps([])

    # Vectorize user preferences
    user_pref_vector = vectorizer.transform([user_preferences])
    user_pref_scores = cosine_similarity(vectorizer.transform([post["hashtags"] for post in all_polls_hashtags_list]), user_pref_vector)

    # Sort posts by relevance to user preferences
    posts_sorted = sorted(zip(all_polls_hashtags_list, user_pref_scores), key=lambda x: x[1], reverse=True)

    # Example: Get top posts from the most relevant cluster
    top_cluster = kmeans.predict(user_pref_vector)[0]
    top_posts_in_cluster = [post for post, score in posts_sorted if kmeans.predict(vectorizer.transform([post["hashtags"]])) == top_cluster]


    # Get top 5 posts
    top_n_posts = []
    n = no_of_polls
    for post in top_posts_in_cluster[:n]:
        post_data = db.collection('polls').document(post['id']).get().to_dict()
        top_n_posts.append(post['id'])

    return json.dumps(top_n_posts)


