# app.py
"""
Single-file AI-Powered Mental Health Support Chatbot.

- Backend: Flask + SQLite (SQLAlchemy)
- Frontend: HTML + CSS + JS served by Flask
- Charts: Chart.js (CDN)
- NLP: TextBlob sentiment
- Safety: crisis keyword detection, no medical diagnosis
"""

import os
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from textblob import TextBlob

# ---------------------------------------------------------------------
# Flask + DB setup
# ---------------------------------------------------------------------

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mental_health_chatbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
db = SQLAlchemy(app)

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
# Services: sentiment, suggestions, crisis detection
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
]


def analyze_sentiment(text: str) -> float:
    if not text:
        return 0.0
    blob = TextBlob(text)
    return float(blob.sentiment.polarity)


def generate_suggestions(mood, stress_level, anxiety_level, sleep_hours, sentiment_score):
    suggestions = []

    if mood in ["sad", "anxious", "stressed"] or (sentiment_score is not None and sentiment_score < 0):
        suggestions.append(
            "It sounds like today may feel heavy. Try pausing for a moment and taking 3 slow deep breaths."
        )
        suggestions.append(
            "If it helps, write a little more about what is bothering you so you can notice patterns in your thoughts."
        )

    if stress_level is not None and stress_level >= 7:
        suggestions.append(
            "Your stress level seems high. A short walk, a glass of water, or a 5-minute break may help reduce tension."
        )

    if anxiety_level is not None and anxiety_level >= 7:
        suggestions.append(
            "Anxiety can feel overwhelming. Try grounding yourself by noticing 5 things you can see and 4 things you can touch."
        )

    if sleep_hours is not None and sleep_hours < 6:
        suggestions.append(
            "Your sleep looks lower than ideal. A simple bedtime routine and reducing screen time before bed may support better rest."
        )

    if mood == "happy" and (stress_level is not None and stress_level <= 4):
        suggestions.append(
            "You seem to be doing relatively well today. Try to keep up the habits that are helping you feel balanced."
        )

    if not suggestions:
        suggestions.append(
            "Thank you for checking in today. Keep noticing how you feel and give yourself space to rest when needed."
        )

    suggestions.append(
        "This chatbot offers emotional wellness support only and is not a substitute for professional mental health care."
    )

    return suggestions


def detect_crisis(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in CRISIS_KEYWORDS)


def crisis_response():
    return {
        "is_crisis": True,
        "message": (
            "It sounds like you may be in immediate emotional distress. "
            "This chatbot cannot provide crisis support. Please contact emergency services, "
            "a crisis helpline, or a trusted person right away."
        ),
        "resources": [
            "If you are in immediate danger, call your local emergency services now.",
            "Reach out to a trusted friend, family member, or licensed mental health professional immediately.",
        ],
    }

# ---------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------

@app.route("/api/health")
def health_check():
    return {"status": "ok", "message": "Mental Health Chatbot API running"}


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

    suggestions = generate_suggestions(
        mood, stress_level, anxiety_level, sleep_hours, sentiment_score
    )

    bot_reply = {
        "is_crisis": False,
        "sentiment_score": sentiment_score,
        "suggestions": suggestions,
        "acknowledgement": "Thank you for sharing how you are feeling today. I am here to offer gentle support.",
        "created_at": datetime.utcnow().isoformat(),
    }
    return jsonify(bot_reply), 200


@app.route("/api/tracking/summary", methods=["GET"])
def get_summary():
    username = request.args.get("username", "guest")
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"entries": []}), 200

    entries = (
        WellnessEntry.query.filter_by(user_id=user.id)
        .order_by(WellnessEntry.created_at.asc())
        .all()
    )
    data = [e.to_dict() for e in entries]
    return jsonify({"entries": data}), 200

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
            max-width: 1250px;
            margin: 0 auto;
            padding: 28px 18px 34px;
        }

        header {
            text-align: center;
            margin-bottom: 24px;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 8px;
            font-weight: 700;
        }

        .subtitle {
            font-size: 1rem;
            color: var(--text-muted);
            margin-bottom: 10px;
        }

        .disclaimer {
            max-width: 760px;
            margin: 0 auto;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 14px;
            font-size: 0.96rem;
            color: #36485f;
            box-shadow: 0 6px 20px rgba(34, 48, 70, 0.06);
        }

        main {
            display: grid;
            grid-template-columns: 1.9fr 1fr;
            gap: 20px;
            align-items: stretch;
        }

        .chat-section,
        .dashboard-section {
            background: var(--card-bg);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 22px;
        }

        .section-title {
            font-size: 1.35rem;
            margin-bottom: 16px;
            font-weight: 700;
        }

        .user-info,
        .field-group {
            margin-bottom: 14px;
        }

        label {
            display: block;
            margin-bottom: 7px;
            font-size: 0.96rem;
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
            padding: 12px 14px;
            font-size: 0.98rem;
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
            grid-template-columns: repeat(2, 1fr);
            gap: 14px;
            margin-bottom: 18px;
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
            font-size: 0.92rem;
            color: var(--text-muted);
        }

        .chat-window {
            height: 340px;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(120, 149, 178, 0.18);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 14px;
        }

        .empty-chat {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.95rem;
            padding: 40px 10px;
        }

        .message {
            margin-bottom: 14px;
            display: flex;
            flex-direction: column;
            max-width: 82%;
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
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            padding: 0 4px;
        }

        .bubble {
            border-radius: 16px;
            padding: 12px 14px;
            line-height: 1.5;
            font-size: 0.96rem;
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

        .input-label {
            margin-bottom: 8px;
            font-size: 0.96rem;
            font-weight: 600;
        }

        .chat-input-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 12px;
            align-items: end;
        }

        .chat-input-row textarea {
            min-height: 72px;
            resize: vertical;
        }

        .send-btn {
            border: none;
            border-radius: 14px;
            background: var(--primary);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            padding: 14px 22px;
            cursor: pointer;
            min-width: 110px;
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
            font-size: 0.88rem;
            color: #b33a3a;
            min-height: 18px;
        }

        .dashboard-top-note {
            font-size: 0.92rem;
            color: var(--text-muted);
            margin-bottom: 12px;
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(120, 149, 178, 0.16);
            border-radius: 18px;
            padding: 14px;
            margin-bottom: 16px;
        }

        .chart-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .no-data-message {
            display: none;
            font-size: 0.92rem;
            color: var(--text-muted);
            margin-bottom: 14px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.55);
            border-radius: 12px;
        }

        @media (max-width: 980px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 640px) {
            header h1 {
                font-size: 2rem;
            }

            .wellness-grid {
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
                    <span class="chat-subtitle">Your messages and supportive responses will appear below.</span>
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

        function addMessage(sender, text, type = "bot", timestamp = null) {
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
                { created_at: new Date(Date.now() - 4 * 86400000).toISOString(), stress_level: 6, sleep_hours: 6.5, mood: "stressed" },
                { created_at: new Date(Date.now() - 3 * 86400000).toISOString(), stress_level: 5, sleep_hours: 7.0, mood: "neutral" },
                { created_at: new Date(Date.now() - 2 * 86400000).toISOString(), stress_level: 7, sleep_hours: 5.5, mood: "anxious" },
                { created_at: new Date(Date.now() - 1 * 86400000).toISOString(), stress_level: 4, sleep_hours: 7.5, mood: "happy" }
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
                    addMessage("Crisis Alert", data.message, "alert");
                    if (data.resources) {
                        data.resources.forEach((resource) => {
                            addMessage("Emergency Support", resource, "alert");
                        });
                    }
                } else {
                    addMessage("Support Bot", data.acknowledgement, "bot", data.created_at);
                    if (data.suggestions) {
                        data.suggestions.forEach((suggestion) => {
                            addMessage("Support Bot", suggestion, "bot", data.created_at);
                        });
                    }
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
                        pointRadius: 4,
                        pointHoverRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            min: minY,
                            ...(maxY !== null ? { max: maxY } : {})
                        }
                    }
                }
            });
        }

        async function loadDashboard() {
            const username = usernameInput.value.trim() || "guest";

            try {
                const res = await fetch(`${API_BASE}/api/tracking/summary?username=${encodeURIComponent(username)}`);
                const data = await res.json();

                let entries = data.entries || [];
                let usingSampleData = false;

                if (!entries.length) {
                    entries = getSampleEntries();
                    usingSampleData = true;
                    noDataMessage.style.display = "block";
                } else {
                    noDataMessage.style.display = "none";
                }

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

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)