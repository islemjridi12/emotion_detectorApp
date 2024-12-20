# Load Database Packages
import sqlite3
# Creates database in the directory
conn = sqlite3.connect('./data/data.db', check_same_thread = False)
c = conn.cursor()


# Function to Track Input & Prediction
def create_emotionclf_table():
	c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TIMESTAMP)')

def add_prediction_details(rawtext, prediction, probability, timeOfvisit):
	c.execute('INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit) VALUES(?, ?, ?, ?)',(rawtext, prediction, probability, timeOfvisit))
	conn.commit()

def view_all_prediction_details():
	c.execute('SELECT * FROM emotionclfTable')
	data = c.fetchall()
	return data