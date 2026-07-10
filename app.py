from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "FedEx Demand Prediction API is Running Successfully!"

if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="127.0.0.1", port=5000, debug=True)