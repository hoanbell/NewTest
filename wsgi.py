from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "<h1>Welcome to Summoner's Drift</h1>"


if __name__ == '__main__':
    app.run(debug=1, host='0.0.0.0')
