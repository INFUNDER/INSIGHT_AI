from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/course')
def course():
    return render_template('course.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')

if __name__ == '__main__':
    app.run(debug=True)
