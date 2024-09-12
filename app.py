from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_questions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class AmazonQues(db.Model):
    q_num = db.Column(db.Integer, primary_key=True, autoincrement=True)
    questions = db.Column(db.String(250), nullable=False)
    option_A = db.Column(db.String(250), nullable=False)
    option_B = db.Column(db.String(250), nullable=False)
    option_C = db.Column(db.String(250), nullable=False)
    option_D = db.Column(db.String(250), nullable=False)
    Answer = db.Column(db.String(2), nullable=False)

class MicrosoftQues(db.Model):
    q_num = db.Column(db.Integer, primary_key=True, autoincrement=True)
    questions = db.Column(db.String(250), nullable=False)
    option_A = db.Column(db.String(250), nullable=False)
    option_B = db.Column(db.String(250), nullable=False)
    option_C = db.Column(db.String(250), nullable=False)
    option_D = db.Column(db.String(250), nullable=False)
    Answer = db.Column(db.String(2), nullable=False)

with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/course')
def course():
    return render_template('course.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')


@app.route('/coursedecp')
def coursedecp():
    amazon_questions = AmazonQues.query.all()
    return render_template('coursedecp.html', questions=amazon_questions)

@app.route('/coursedecp2')
def coursedecp2():
    amazon_questions = AmazonQues.query.all()
    return render_template('coursedecp2.html', questions=amazon_questions)

@app.route('/coursedecp3')
def coursedecp3():
    amazon_questions = AmazonQues.query.all()
    return render_template('coursedecp3.html', questions=amazon_questions)

@app.route('/aiinterview')
def aiinterview():
    return render_template('aiinterview.html')

if __name__ == '__main__':
    app.run(debug=True)
