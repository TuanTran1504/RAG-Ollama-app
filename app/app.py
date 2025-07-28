from flask import Flask, render_template, request
import asyncio
from main import generate_responses

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = ""

    if request.method == "POST":
        question = request.form["question"]
        try:
            responses = asyncio.run(generate_responses([question]))
            answer = responses[0]["answer"] if responses else "No answer generated."
        except Exception as e:
            answer = f"Error: {str(e)}"

    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)