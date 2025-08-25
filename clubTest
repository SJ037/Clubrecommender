from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

CLUBS = [
    {
        "name": "Robotics Club",
        "description": "Explore engineering and programming by building robots and competing in challenges."
    },
    {
        "name": "Drama Club",
        "description": "Practice acting, theatre, and performance skills through plays and improv sessions."
    },
    {
        "name": "Environmental Club",
        "description": "Promote sustainability, climate action, and nature conservation projects."
    },
    {
        "name": "Debate Club",
        "description": "Hone public speaking, logical reasoning, and political discussion skills."
    },
    {
        "name": "Art Club",
        "description": "Express creativity through drawing, painting, and various art projects."
    },
    {
        "name": "Gaming Club",
        "description": "Gather to play video games, discuss gaming strategies, and organize tournaments."
    },
    {
        "name": "Math Club",
        "description": "Engage in problem solving, mathematical challenges, and competitions."
    },
    {
        "name": "Business Club",
        "description": "Learn about entrepreneurship, finance, and leadership through projects and guest speakers."
    },
]

def get_club_recommendations(user_likes, user_dislikes):
    likes_text = " ".join(user_likes)
    dislikes_text = " ".join(user_dislikes) if user_dislikes else ""

    likes_embedding = model.encode(likes_text, convert_to_tensor=True)
    dislikes_embedding = model.encode(dislikes_text, convert_to_tensor=True) if dislikes_text else None

    scored_clubs = []
    for club in CLUBS:
        desc_embedding = model.encode(club["description"], convert_to_tensor=True)
        like_score = util.cos_sim(likes_embedding, desc_embedding).item()
        dislike_score = util.cos_sim(dislikes_embedding, desc_embedding).item() if dislikes_embedding is not None else 0
        score = like_score - dislike_score
        scored_clubs.append((club["name"], score))

    scored_clubs.sort(key=lambda x: x[1], reverse=True)
    return scored_clubs[:3]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        likes_input = request.form.get("likes", "")
        dislikes_input = request.form.get("dislikes", "")
        user_likes = [w.strip() for w in likes_input.split(",") if w.strip()]
        user_dislikes = [w.strip() for w in dislikes_input.split(",") if w.strip()]
        recommendations = get_club_recommendations(user_likes, user_dislikes)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)