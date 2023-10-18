from flask import Flask, request, render_template
from chatbotscript.py import chatbot_response  # Import your chatbot function

app = Flask(__name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form.get("user_input")
    response = chatbot_response(user_input)  # Call your chatbot function
    return str(response)

if __name__ == "__main__":
    app.run()