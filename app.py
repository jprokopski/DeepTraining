from flask import Flask, Response, render_template, redirect
import cv2
import functions

app = Flask(__name__)

cap = cv2.VideoCapture(0)


@app.route('/')
def index():
    cap.release()
    return render_template('index.html')

@app.route('/biceps')
def show_biceps():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return Response(functions.biceps(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/przysiad')
def show_przysiad():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return Response(functions.przysiad(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pompka')
def show_pompka():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return Response(functions.pompka(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/brzuszki')
def show_brzuszki():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return Response(functions.brzuszki(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/military')
def show_military():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return Response(functions.military(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/wznosy')
def show_wznosy():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return Response(functions.wznosy(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)
