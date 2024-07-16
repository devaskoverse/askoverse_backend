import firebase_admin
from firebase_admin import firestore, storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import datetime
import tempfile

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
        'vectorizer': vectorizer
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
