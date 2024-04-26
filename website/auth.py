# this file is to route the login and sign up page 
# blue_print is used to modularize your Flask applications, making them easier to maintain and scale.
# we use werkzeug .security for password securitry,generating hash password and verfication of these passwords
# we use flask_login for user status 
# flash message give the pop up message 

from flask import Blueprint, render_template, request, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User # callindg the class User
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST']) # routing to login page
def login():
    if request.method == 'POST':               # getting the email and password from the form
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first() # creates an object user
        if user:
            if check_password_hash(user.password, password): # checking the password
                flash('Logged IN!', category='success')   
                login_user(user, remember=True)
                return redirect(url_for('views.home'))    # after successful login user is redicted to home page
            else:
                flash('Incorrect password, try again.', category='error') # incorrect password error
        else:
            flash('Email not found', category='error')  # incorrect email error

    return render_template("login.html", user=current_user) # render the login page

@auth.route('/logout') # routing logout page
@login_required
def logout():
    logout_user()  # user status log out
    return redirect(url_for('auth.login')) # redirceting to login page


@auth.route('/sign-up', methods=['GET', 'POST'])  # routing to sign up page
def sign_up():
    if request.method == 'POST':           # getting inputs from the form
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()  # conditions for creating a account 
        if user:                                             
            flash('Email already exists.', category='error')
        elif len(email) < 5:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 3:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 5:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1))
            db.session.add(new_user)  # adding the new user to the database
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return redirect(url_for('views.home'))   # after successful sign-up ,redirect to home page

    return render_template("sign_up.html", user=current_user)  # rendering the sign-up page
