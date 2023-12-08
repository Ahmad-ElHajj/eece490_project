import threading
from flask import Blueprint, render_template, session, request, redirect, url_for
from .UserManager import users
from model.ImageDetector import get_pesticide
import random, os,shutil

views = Blueprint('views', __name__)

def handle_user():
    if 'token' not in session:
        session['token'] = ''.join(str(random.randint(0,9)) for _ in range(64))
    users.getUser(session['token'])
    users.save()

def current_user():
    return users.getUser(session['token'])

@views.route('/', methods = ['GET', 'POST'])
def home():
    handle_user()
    user = current_user()
    if request.method == "POST":
        directory_path = os.path.join('temp',user.token)
        if shutil.os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        for image in request.files.getlist("images[]"):
            image_path = os.path.join(directory_path,image.filename)
            image.save(image_path)
        
        threading.Thread(target=get_pesticide, args=(current_user(),)).start()
        return redirect(url_for('views.processing'))

    return render_template("home.html")

@views.route('/meetus')
def meetus():
    return render_template("team.html")

@views.route('/report')
def report():
    return render_template("report.html")


@views.route("/processing")
def processing():
    if current_user().is_processing:
        return render_template("processing.html")
    return redirect(url_for("views.results"))

@views.route("results", methods = ['GET', 'POST'])
def results():
    if request.method=="POST":
        for e in request.form.get("selectedPesticides"):
            current_user().history_of_purchases[e] = current_user().history_of_purchases[e]+1 if e in current_user().history_of_purchases else 1

    return render_template("results.html", pesticides_dict = current_user().pesticide_pests)

@views.route('/deb')
def deb():
    print(users.display())
    return "<p>things done.</p>"
