from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
