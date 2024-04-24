# importing th database from init.py
# User_Mixin used in conjunction with a user model class to add common functionality such as user authentication, authorization, and session management
# flask_wtf and wftforms for simplifed flask command for getting image from website
# FileField is used to handle the file upload
# SubmitField is used submit button for uploading the image

from . import db
from flask_login import UserMixin
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField


class User(db.Model, UserMixin): # creating class ,user for the database
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

