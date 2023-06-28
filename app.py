from flask import Flask, render_template
import csv
import subprocess

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start')
def start():
    subprocess.Popen(['python', 'main.py'])
    return "Main script executed successfully."

i=0
@app.route('/data')
def data():
    try:
        global i
        i+=1
        with open('car_positions.csv', 'r') as file:
            reader = csv.reader(file)
            data_list = list(reader)
            # Get the latest data from the last row of the CSV file
            latest_data = data_list[i]
            frame_no = latest_data[0]
            cx = latest_data[1]
            cy = latest_data[2]
            color_name = latest_data[3]
            return {'frame_no': frame_no, 'cx': cx, 'cy': cy, 'color_name': color_name}
    except:
        return {'frame_no': -1}  # suppress the error for not returning valid data


if __name__ == '__main__':
    app.run(debug=True)
