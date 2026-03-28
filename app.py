# app.py
"""
Single-file AI-Powered Mental Wellness Support Chatbot
Dynamic AI version using OpenAI Responses API

Run:
1. pip install -r requirements.txt
2. Set OPENAI_API_KEY
3. python app.py
"""

import os
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from textblob import TextBlob
from openai import OpenAI

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mental_health_chatbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
db = SQLAlchemy(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

    entries = db.relationship("WellnessEntry", backref="user", lazy=True)

    def to_dict(self):
        return {"id": self.id, "username": self.username}


class WellnessEntry(db.Model):
    __tablename__ = "wellness_entries"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    mood = db.Column(db.String(20), nullable=True)
    stress_level = db.Column(db.Integer, nullable=True)
    anxiety_level = db.Column(db.Integer, nullable=True)
    sleep_hours = db.Column(db.Float, nullable=True)
    journal_text = db.Column(db.Text, nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "mood": self.mood,
            "stress_level": self.stress_level,
            "anxiety_level": self.anxiety_level,
            "sleep_hours": self.sleep_hours,
            "journal_text": self.journal_text,
            "sentiment_score": self.sentiment_score,
            "created_at": self.created_at.isoformat(),
        }


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

CRISIS_KEYWORDS = [
    "i want to die",
    "i want to hurt myself",
    "i don't want to live",
    "i dont want to live",
    "kill myself",
    "end my life",
    "self harm",
    "hurt myself",
    "suicide",
]

MOOD_SCORE_MAP = {
    "sad": 2,
    "anxious": 3,
    "stressed": 4,
    "neutral": 6,
    "happy": 9,
}


def analyze_sentiment(text: str) -> float:
    if not text:
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def sentiment_label(score: float) -> str:
    if score > 0.2:
        return "Positive"
    if score < -0.2:
        return "Negative"
    return "Neutral"


def detect_crisis(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in CRISIS_KEYWORDS)


def crisis_response():
    return {
        "is_crisis": True,
        "message": (
            "⚠️ It sounds like you may be in immediate emotional distress.\n\n"
            "This chatbot cannot provide crisis support. Please contact emergency services, "
            "a crisis helpline, or a trusted person right away."
        ),
        "resources": [
            "Call your local emergency services if you are in immediate danger.",
            "Reach out to a trusted friend, family member, or licensed mental health professional now.",
        ],
        "created_at": datetime.utcnow().isoformat(),
    }


def build_fallback_reply(mood, stress_level, anxiety_level, sleep_hours, sentiment_score):
    detected = sentiment_label(sentiment_score)

    if mood in ["sad", "anxious", "stressed"] or sentiment_score < 0:
        text = (
            "Thank you for sharing that with me. It sounds like today feels a bit heavy. "
            "Try taking a small pause, drinking some water, and focusing on one gentle step at a time. "
            "If you want, you can tell me what is bothering you most right now."
        )
    elif mood == "happy" or sentiment_score > 0.2:
        text = (
            "I am really glad to hear that. It sounds like something positive is part of your day. "
            "Try to hold onto that feeling and notice what helped create it."
        )
    else:
        text = (
            "Thank you for checking in. I am here with you. "
            "Try to slow down for a moment and notice what your mind and body need right now."
        )

    extra_parts = []

    if stress_level is not None and stress_level >= 7:
        extra_parts.append("Your stress level seems high, so a short break or a small walk may help.")
    if anxiety_level is not None and anxiety_level >= 7:
        extra_parts.append("Your anxiety looks elevated, so grounding yourself with your surroundings may help.")
    if sleep_hours is not None and sleep_hours < 6:
        extra_parts.append("You also mentioned lower sleep, which can make everything feel harder.")

    if extra_parts:
        text += " " + " ".join(extra_parts)

    text += " This chatbot offers emotional wellness support only and is not a substitute for professional mental health care."
    return text, detected


def build_ai_prompt(message, mood, stress_level, anxiety_level, sleep_hours, detected_sentiment):
    return f"""
You are a supportive emotional wellness chatbot.

Rules:
- Be warm, empathetic, natural, and concise.
- Respond specifically to the user's exact message.
- Do not repeat the same generic response.
- Mention mood/stress/anxiety/sleep details only if relevant.
- Do not diagnose any condition.
- Do not say you are a therapist.
- Do not give dangerous or extreme advice.
- If the user sounds happy, respond positively and naturally.
- If the user sounds sad, stressed, lonely, anxious, or overwhelmed, respond gently and practically.
- Ask at most one short follow-up question.
- Keep the reply under 100 words.
- Do not use bullet points.

User data:
Mood: {mood or "Not provided"}
Stress level: {stress_level if stress_level is not None else "Not provided"}
Anxiety level: {anxiety_level if anxiety_level is not None else "Not provided"}
Sleep hours: {sleep_hours if sleep_hours is not None else "Not provided"}
Detected sentiment: {detected_sentiment}

User message:
{message}
""".strip()


def generate_dynamic_ai_reply(message, mood, stress_level, anxiety_level, sleep_hours, sentiment_score):
    detected = sentiment_label(sentiment_score)

    if not client:
        return build_fallback_reply(mood, stress_level, anxiety_level, sleep_hours, sentiment_score)

    prompt = build_ai_prompt(
        message=message,
        mood=mood,
        stress_level=stress_level,
        anxiety_level=anxiety_level,
        sleep_hours=sleep_hours,
        detected_sentiment=detected,
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        text = (response.output_text or "").strip()
        if not text:
            return build_fallback_reply(mood, stress_level, anxiety_level, sleep_hours, sentiment_score)
        return text, detected
    except Exception:
        return build_fallback_reply(mood, stress_level, anxiety_level, sleep_hours, sentiment_score)


def get_weekly_summary(entries):
    if not entries:
        return {
            "average_stress": 0,
            "average_sleep": 0,
            "average_anxiety": 0,
            "most_common_mood": "No data",
            "checkins": 0,
        }

    stress_values = [e.stress_level for e in entries if e.stress_level is not None]
    sleep_values = [e.sleep_hours for e in entries if e.sleep_hours is not None]
    anxiety_values = [e.anxiety_level for e in entries if e.anxiety_level is not None]
    moods = [e.mood for e in entries if e.mood]

    def avg(values):
        return round(sum(values) / len(values), 1) if values else 0

    most_common_mood = "No data"
    if moods:
        mood_counts = {}
        for mood in moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        most_common_mood = max(mood_counts, key=mood_counts.get).title()

    return {
        "average_stress": avg(stress_values),
        "average_sleep": avg(sleep_values),
        "average_anxiety": avg(anxiety_values),
        "most_common_mood": most_common_mood,
        "checkins": len(entries),
    }


# ---------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------

@app.route("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "Mental Health Chatbot API running",
        "openai_configured": bool(client),
        "model": OPENAI_MODEL,
    }


@app.route("/api/chat/message", methods=["POST"])
def handle_message():
    data = request.get_json() or {}

    username = (data.get("username") or "guest").strip()
    message = (data.get("message") or "").strip()
    mood = data.get("mood")
    stress_level = data.get("stress_level")
    anxiety_level = data.get("anxiety_level")
    sleep_hours = data.get("sleep_hours")

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username)
        db.session.add(user)
        db.session.commit()

    if detect_crisis(message):
        return jsonify(crisis_response()), 200

    sentiment_score = analyze_sentiment(message)

    entry = WellnessEntry(
        user_id=user.id,
        mood=mood,
        stress_level=stress_level,
        anxiety_level=anxiety_level,
        sleep_hours=sleep_hours,
        journal_text=message,
        sentiment_score=sentiment_score,
    )
    db.session.add(entry)
    db.session.commit()

    ai_reply, detected_sentiment = generate_dynamic_ai_reply(
        message=message,
        mood=mood,
        stress_level=stress_level,
        anxiety_level=anxiety_level,
        sleep_hours=sleep_hours,
        sentiment_score=sentiment_score,
    )

    return jsonify({
        "is_crisis": False,
        "sentiment_score": sentiment_score,
        "detected_sentiment": detected_sentiment,
        "grouped_message": ai_reply,
        "created_at": datetime.utcnow().isoformat(),
    }), 200


@app.route("/api/tracking/summary", methods=["GET"])
def get_summary():
    username = request.args.get("username", "guest")
    user = User.query.filter_by(username=username).first()

    if not user:
        return jsonify({
            "entries": [],
            "weekly_summary": {
                "average_stress": 0,
                "average_sleep": 0,
                "average_anxiety": 0,
                "most_common_mood": "No data",
                "checkins": 0,
            }
        }), 200

    entries = (
        WellnessEntry.query.filter_by(user_id=user.id)
        .order_by(WellnessEntry.created_at.asc())
        .all()
    )

    one_week_ago = datetime.utcnow() - timedelta(days=7)
    weekly_entries = [e for e in entries if e.created_at >= one_week_ago]

    return jsonify({
        "entries": [entry.to_dict() for entry in entries],
        "weekly_summary": get_weekly_summary(weekly_entries),
    }), 200


# ---------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mental Wellness Support Chatbot</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-start: #a8edea;
            --bg-end: #7fb3ff;
            --card-bg: rgba(255, 255, 255, 0.78);
            --text-main: #223046;
            --text-muted: #5b6b7f;
            --border: rgba(120, 149, 178, 0.22);
            --shadow: 0 12px 30px rgba(34, 48, 70, 0.12);
            --primary: #4c8fe8;
            --primary-hover: #3b78c9;
            --user-bubble: #d9ecff;
            --bot-bubble: #e5f7ea;
            --alert-bg: #fff1f1;
            --alert-border: #e45b5b;
            --input-bg: rgba(255, 255, 255, 0.92);
            --summary-bg: rgba(255, 255, 255, 0.58);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%);
            color: var(--text-main);
            min-height: 100vh;
        }

        .app-container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 24px 16px 30px;
        }

        header {
            text-align: center;
            margin-bottom: 18px;
        }

        header h1 {
            font-size: 2.2rem;
            margin-bottom: 8px;
            font-weight: 700;
        }

        .subtitle {
            font-size: 0.98rem;
            color: var(--text-muted);
            margin-bottom: 10px;
        }

        .disclaimer {
            max-width: 760px;
            margin: 0 auto;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 14px;
            font-size: 0.9rem;
            color: #36485f;
            box-shadow: 0 6px 20px rgba(34, 48, 70, 0.06);
        }

        main {
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 18px;
            align-items: start;
        }

        .chat-section,
        .dashboard-section {
            background: var(--card-bg);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 18px;
        }

        .section-title {
            font-size: 1.25rem;
            margin-bottom: 14px;
            font-weight: 700;
        }

        .user-info,
        .field-group {
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 6px;
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text-main);
        }

        input,
        select,
        textarea {
            width: 100%;
            border: 1px solid #c9d5e4;
            background: var(--input-bg);
            border-radius: 12px;
            padding: 11px 13px;
            font-size: 0.95rem;
            color: var(--text-main);
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input:focus,
        select:focus,
        textarea:focus {
            border-color: #7aaaf1;
            box-shadow: 0 0 0 4px rgba(76, 143, 232, 0.12);
        }

        .wellness-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 16px;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .chat-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin: 6px 0 10px;
            flex-wrap: wrap;
        }

        .chat-subtitle {
            font-size: 0.88rem;
            color: var(--text-muted);
        }

        .chat-window {
            height: 330px;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(120, 149, 178, 0.18);
            border-radius: 18px;
            padding: 14px;
            margin-bottom: 12px;
        }

        .empty-chat {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            padding: 36px 10px;
        }

        .message {
            margin-bottom: 12px;
            display: flex;
            flex-direction: column;
            max-width: 84%;
        }

        .message.user {
            margin-left: auto;
            align-items: flex-end;
        }

        .message.bot,
        .message.alert {
            margin-right: auto;
            align-items: flex-start;
        }

        .message-meta {
            font-size: 0.76rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            padding: 0 4px;
        }

        .bubble {
            border-radius: 16px;
            padding: 11px 13px;
            line-height: 1.55;
            font-size: 0.94rem;
            box-shadow: 0 4px 12px rgba(34, 48, 70, 0.05);
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .message.user .bubble {
            background: var(--user-bubble);
            border-bottom-right-radius: 6px;
        }

        .message.bot .bubble {
            background: var(--bot-bubble);
            border-bottom-left-radius: 6px;
        }

        .message.alert .bubble {
            background: var(--alert-bg);
            border: 1px solid var(--alert-border);
            color: #8e1f1f;
            border-bottom-left-radius: 6px;
        }

        .sentiment-chip {
            display: inline-block;
            margin-top: 8px;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(76, 143, 232, 0.12);
            color: #24589d;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .input-label {
            margin-bottom: 8px;
            font-size: 0.92rem;
            font-weight: 600;
        }

        .chat-input-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
            align-items: end;
        }

        .chat-input-row textarea {
            min-height: 64px;
            resize: vertical;
        }

        .send-btn {
            border: none;
            border-radius: 14px;
            background: var(--primary);
            color: white;
            font-size: 0.96rem;
            font-weight: 600;
            padding: 13px 22px;
            cursor: pointer;
            min-width: 100px;
            transition: background 0.2s ease, transform 0.15s ease;
        }

        .send-btn:hover {
            background: var(--primary-hover);
        }

        .send-btn:active {
            transform: translateY(1px);
        }

        .validation-msg {
            margin-top: 8px;
            font-size: 0.86rem;
            color: #b33a3a;
            min-height: 18px;
        }

        .dashboard-top-note {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-bottom: 10px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }

        .summary-card {
            background: var(--summary-bg);
            border: 1px solid rgba(120, 149, 178, 0.16);
            border-radius: 16px;
            padding: 12px;
        }

        .summary-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 5px;
        }

        .summary-value {
            font-size: 1.08rem;
            font-weight: 700;
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(120, 149, 178, 0.16);
            border-radius: 18px;
            padding: 12px;
            margin-bottom: 12px;
        }

        .chart-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .chart-card canvas {
            width: 100% !important;
            height: 170px !important;
        }

        .no-data-message {
            display: none;
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-bottom: 10px;
            padding: 9px 11px;
            background: rgba(255, 255, 255, 0.55);
            border-radius: 12px;
        }

        @media (max-width: 980px) {
            main {
                grid-template-columns: 1fr;
            }

            .dashboard-section {
                order: 2;
            }

            .chat-section {
                order: 1;
            }
        }

        @media (max-width: 640px) {
            header h1 {
                font-size: 1.8rem;
            }

            .wellness-grid,
            .summary-grid {
                grid-template-columns: 1fr;
            }

            .chat-input-row {
                grid-template-columns: 1fr;
            }

            .send-btn {
                width: 100%;
            }

            .message {
                max-width: 100%;
            }

            .chart-card canvas {
                height: 160px !important;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header>
            <h1>Mental Wellness Support Chatbot</h1>
            <p class="subtitle">Track your mood, reflect on your thoughts, and view your wellness trends in one place.</p>
            <p class="disclaimer">
                This chatbot is for emotional wellness support only and is not a substitute for professional medical advice or crisis care.
            </p>
        </header>

        <main>
            <section class="chat-section">
                <h2 class="section-title">Wellness Check-In</h2>

                <div class="user-info">
                    <label for="username">Username (optional)</label>
                    <input type="text" id="username" placeholder="Enter a name or continue as guest" />
                </div>

                <div class="wellness-grid">
                    <div class="field-group full-width">
                        <label for="mood">Mood</label>
                        <select id="mood">
                            <option value="">Select your mood</option>
                            <option value="happy">Happy</option>
                            <option value="sad">Sad</option>
                            <option value="anxious">Anxious</option>
                            <option value="stressed">Stressed</option>
                            <option value="neutral">Neutral</option>
                        </select>
                    </div>

                    <div class="field-group">
                        <label for="stress">Stress Level (1-10)</label>
                        <input type="number" id="stress" min="1" max="10" placeholder="e.g. 6" />
                    </div>

                    <div class="field-group">
                        <label for="anxiety">Anxiety Level (1-10)</label>
                        <input type="number" id="anxiety" min="1" max="10" placeholder="e.g. 5" />
                    </div>

                    <div class="field-group full-width">
                        <label for="sleep">Sleep Duration (hours)</label>
                        <input type="number" id="sleep" min="0" max="24" step="0.5" placeholder="e.g. 7.5" />
                    </div>
                </div>

                <div class="chat-header-row">
                    <h3>Chat Conversation</h3>
                    <span class="chat-subtitle">Your messages and AI responses will appear below.</span>
                </div>

                <div class="chat-window" id="chatWindow">
                    <div class="empty-chat" id="emptyChat">
                        Start by entering how you feel today. Your conversation will appear here.
                    </div>
                </div>

                <div>
                    <label class="input-label" for="messageInput">How are you feeling today?</label>
                    <div class="chat-input-row">
                        <textarea id="messageInput" placeholder="Share your thoughts, stress, mood, or anything on your mind..."></textarea>
                        <button id="sendBtn" class="send-btn">Send</button>
                    </div>
                    <div class="validation-msg" id="validationMsg"></div>
                </div>
            </section>

            <section class="dashboard-section">
                <h2 class="section-title">Your Wellness Trends</h2>
                <p class="dashboard-top-note">
                    Your recent check-ins are shown below. If you have not added entries yet, sample trend data will be displayed.
                </p>

                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-label">Average Stress</div>
                        <div class="summary-value" id="avgStress">0</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Average Sleep</div>
                        <div class="summary-value" id="avgSleep">0 hrs</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Average Anxiety</div>
                        <div class="summary-value" id="avgAnxiety">0</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Most Common Mood</div>
                        <div class="summary-value" id="commonMood">No data</div>
                    </div>
                </div>

                <div class="no-data-message" id="noDataMessage">
                    No wellness data available yet. Showing sample trend data for layout preview.
                </div>

                <div class="chart-card">
                    <div class="chart-title">Stress Trend</div>
                    <canvas id="stressChart"></canvas>
                </div>

                <div class="chart-card">
                    <div class="chart-title">Sleep Trend</div>
                    <canvas id="sleepChart"></canvas>
                </div>

                <div class="chart-card">
                    <div class="chart-title">Mood Trend</div>
                    <canvas id="moodChart"></canvas>
                </div>
            </section>
        </main>
    </div>

    <script>
        const API_BASE = "";

        const chatWindow = document.getElementById("chatWindow");
        const emptyChat = document.getElementById("emptyChat");
        const sendBtn = document.getElementById("sendBtn");
        const messageInput = document.getElementById("messageInput");
        const usernameInput = document.getElementById("username");
        const moodSelect = document.getElementById("mood");
        const stressInput = document.getElementById("stress");
        const anxietyInput = document.getElementById("anxiety");
        const sleepInput = document.getElementById("sleep");
        const validationMsg = document.getElementById("validationMsg");
        const noDataMessage = document.getElementById("noDataMessage");

        const avgStressEl = document.getElementById("avgStress");
        const avgSleepEl = document.getElementById("avgSleep");
        const avgAnxietyEl = document.getElementById("avgAnxiety");
        const commonMoodEl = document.getElementById("commonMood");

        let stressChart = null;
        let sleepChart = null;
        let moodChart = null;

        function formatTime(dateValue = null) {
            const date = dateValue ? new Date(dateValue) : new Date();
            return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }

        function removeEmptyChatState() {
            if (emptyChat) {
                emptyChat.style.display = "none";
            }
        }

        function addMessage(sender, text, type = "bot", timestamp = null, detectedSentiment = null) {
            removeEmptyChatState();

            const wrapper = document.createElement("div");
            wrapper.classList.add("message", type);

            const meta = document.createElement("div");
            meta.classList.add("message-meta");
            meta.textContent = `${sender} • ${formatTime(timestamp)}`;

            const bubble = document.createElement("div");
            bubble.classList.add("bubble");
            bubble.textContent = text;

            wrapper.appendChild(meta);
            wrapper.appendChild(bubble);

            if (detectedSentiment && type === "bot") {
                const chip = document.createElement("div");
                chip.classList.add("sentiment-chip");
                chip.textContent = `Detected Sentiment: ${detectedSentiment}`;
                wrapper.appendChild(chip);
            }

            chatWindow.appendChild(wrapper);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        function showValidation(message) {
            validationMsg.textContent = message || "";
        }

        function getMoodScore(mood) {
            const moodMap = {
                sad: 2,
                anxious: 3,
                stressed: 4,
                neutral: 6,
                happy: 9
            };
            return moodMap[mood] || 5;
        }

        function getSampleEntries() {
            return [
                { created_at: new Date(Date.now() - 4 * 86400000).toISOString(), stress_level: 6, sleep_hours: 6.5, anxiety_level: 5, mood: "stressed" },
                { created_at: new Date(Date.now() - 3 * 86400000).toISOString(), stress_level: 5, sleep_hours: 7.0, anxiety_level: 4, mood: "neutral" },
                { created_at: new Date(Date.now() - 2 * 86400000).toISOString(), stress_level: 7, sleep_hours: 5.5, anxiety_level: 7, mood: "anxious" },
                { created_at: new Date(Date.now() - 1 * 86400000).toISOString(), stress_level: 4, sleep_hours: 7.5, anxiety_level: 3, mood: "happy" }
            ];
        }

        async function sendMessage() {
            const message = messageInput.value.trim();

            if (!message) {
                showValidation("Please enter a message before sending.");
                return;
            }

            showValidation("");

            const payload = {
                username: usernameInput.value.trim() || "guest",
                message: message,
                mood: moodSelect.value || null,
                stress_level: stressInput.value ? parseInt(stressInput.value, 10) : null,
                anxiety_level: anxietyInput.value ? parseInt(anxietyInput.value, 10) : null,
                sleep_hours: sleepInput.value ? parseFloat(sleepInput.value) : null
            };

            addMessage(payload.username, message, "user");
            messageInput.value = "";

            try {
                const res = await fetch(`${API_BASE}/api/chat/message`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();

                if (!res.ok) {
                    showValidation(data.error || "Something went wrong. Please try again.");
                    return;
                }

                if (data.is_crisis) {
                    addMessage("Crisis Alert", data.message, "alert", data.created_at);
                    if (data.resources) {
                        data.resources.forEach((resource) => {
                            addMessage("Emergency Support", resource, "alert", data.created_at);
                        });
                    }
                } else {
                    addMessage("AI Support Bot", data.grouped_message, "bot", data.created_at, data.detected_sentiment);
                }

                await loadDashboard();
            } catch (err) {
                console.error(err);
                addMessage("System", "There was an error connecting to the server.", "alert");
            }
        }

        function destroyCharts() {
            if (stressChart) stressChart.destroy();
            if (sleepChart) sleepChart.destroy();
            if (moodChart) moodChart.destroy();
        }

        function createLineChart(canvasId, label, labels, data, borderColor, bgColor, minY = 0, maxY = null) {
            const ctx = document.getElementById(canvasId).getContext("2d");
            return new Chart(ctx, {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [{
                        label: label,
                        data: data,
                        borderColor: borderColor,
                        backgroundColor: bgColor,
                        fill: true,
                        tension: 0.35,
                        pointRadius: 3,
                        pointHoverRadius: 4,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                boxWidth: 22,
                                font: {
                                    size: 10
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                font: {
                                    size: 10
                                }
                            }
                        },
                        y: {
                            beginAtZero: true,
                            min: minY,
                            ...(maxY !== null ? { max: maxY } : {}),
                            ticks: {
                                font: {
                                    size: 10
                                }
                            }
                        }
                    }
                }
            });
        }

        function updateWeeklySummary(summary) {
            avgStressEl.textContent = summary.average_stress ?? 0;
            avgSleepEl.textContent = `${summary.average_sleep ?? 0} hrs`;
            avgAnxietyEl.textContent = summary.average_anxiety ?? 0;
            commonMoodEl.textContent = summary.most_common_mood || "No data";
        }

        async function loadDashboard() {
            const username = usernameInput.value.trim() || "guest";

            try {
                const res = await fetch(`${API_BASE}/api/tracking/summary?username=${encodeURIComponent(username)}`);
                const data = await res.json();

                let entries = data.entries || [];
                let weeklySummary = data.weekly_summary || {
                    average_stress: 0,
                    average_sleep: 0,
                    average_anxiety: 0,
                    most_common_mood: "No data",
                    checkins: 0
                };

                let usingSampleData = false;

                if (!entries.length) {
                    entries = getSampleEntries();
                    usingSampleData = true;
                    noDataMessage.style.display = "block";

                    weeklySummary = {
                        average_stress: 5.5,
                        average_sleep: 6.6,
                        average_anxiety: 4.8,
                        most_common_mood: "Neutral",
                        checkins: 4
                    };
                } else {
                    noDataMessage.style.display = "none";
                }

                updateWeeklySummary(weeklySummary);

                const labels = entries.map((entry) =>
                    new Date(entry.created_at).toLocaleDateString([], {
                        month: "short",
                        day: "numeric"
                    })
                );

                const stressData = entries.map((entry) => entry.stress_level ?? 0);
                const sleepData = entries.map((entry) => entry.sleep_hours ?? 0);
                const moodData = entries.map((entry) => getMoodScore(entry.mood));

                destroyCharts();

                stressChart = createLineChart(
                    "stressChart",
                    "Stress Level",
                    labels,
                    stressData,
                    "#ff7b94",
                    "rgba(255, 123, 148, 0.18)",
                    0,
                    10
                );

                sleepChart = createLineChart(
                    "sleepChart",
                    "Sleep Hours",
                    labels,
                    sleepData,
                    "#5aa9ff",
                    "rgba(90, 169, 255, 0.18)",
                    0,
                    12
                );

                moodChart = createLineChart(
                    "moodChart",
                    usingSampleData ? "Mood Score (Sample)" : "Mood Score",
                    labels,
                    moodData,
                    "#6bcf9b",
                    "rgba(107, 207, 155, 0.18)",
                    0,
                    10
                );
            } catch (err) {
                console.error(err);
                noDataMessage.style.display = "block";
                noDataMessage.textContent = "Unable to load wellness data right now.";
            }
        }

        sendBtn.addEventListener("click", sendMessage);

        messageInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        document.addEventListener("DOMContentLoaded", () => {
            loadDashboard();
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)