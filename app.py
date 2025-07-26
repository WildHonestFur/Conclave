import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import mysql.connector as m
from flask import Flask, render_template
import mysql.connector

app = Flask(__name__)

def get_top_5_teams():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="****",
        database="conclave"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT name, points FROM leader ORDER BY points DESC LIMIT 5")
    teams = cursor.fetchall()
    conn.close()
    return teams

@app.route('/')
def leaderboard():
    teams = get_top_5_teams()
    teams_dict = [{"TeamName": t[0], "Score": t[1]} for t in teams]
    return render_template('leaderboard.html', teams=teams_dict)

if __name__ == "__main__":
    app.run(debug=True)
