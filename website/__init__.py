# this file instialises the program
from flask import Flask                    # importing the modules
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

db = SQLAlchemy()   # creating a database for login
DB_NAME = "database.db"

def create_app(): # creating the flask application
    app=Flask(__name__)
    app.config['SECRET_KEY'] = 'qazxcvbnm wsxc'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'  
    app.config['UPLOAD_FOLDER']='static/files'
    db.init_app(app)

    from .views import views  # import other python files
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')  # registering the bluprints
    app.register_blueprint(auth, url_prefix='/')

    from .models import User  # importing the user class from model.py
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()         # The login manager contains the code that lets your application and Flask-Login work together,                                       
    login_manager.login_view = 'auth.login' # such as how to load a user from an ID, where to send users when they need to log in, and the like.
    login_manager.init_app(app)

    @login_manager.user_loader       # This callback is used to reload the user object from the user ID stored in the session.
    def load_user(id):               # It should take the str ID of a user, and return the corresponding user object.
        return User.query.get(int(id)) 

    return app

def create_database(app):              # creating the database inside thne flask app
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
