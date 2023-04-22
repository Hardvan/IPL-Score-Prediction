import pickle
import joblib
import numpy as np
from flask import Flask,render_template,request

regressor = joblib.load('iplmodel_ridge.sav')
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',val='')

@app.route('/predict',methods=['POST'])
def predict():

    a = []

    if request.method == 'POST':
        
        # Venue
        venue = request.form['venue']
        venues = ['ACA-VDCA Stadium, Visakhapatnam', 'Barabati Stadium, Cuttack', 
                  'Dr DY Patil Sports Academy, Mumbai', 'Dubai International Cricket Stadium, Dubai',
                  'Eden Gardens, Kolkata', 'Feroz Shah Kotla, Delhi', 
                  'Himachal Pradesh Cricket Association Stadium, Dharamshala',
                  'Holkar Cricket Stadium, Indore', 'JSCA International Stadium Complex, Ranchi', 
                  'M Chinnaswamy Stadium, Bangalore', 'MA Chidambaram Stadium, Chepauk',
                  'Maharashtra Cricket Association Stadium, Pune', 'Punjab Cricket Association Stadium, Mohali', 
                  'Raipur International Cricket Stadium, Raipur', 'Rajiv Gandhi International Stadium, Uppal', 
                  'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium, Jaipur', 
                  'Sharjah Cricket Stadium, Sharjah', 'Sheikh Zayed Stadium, Abu-Dhabi', 'Wankhede Stadium, Mumbai']
        lst = [0] * 20      # [0,0, ... ,0]
        index = venues.index(venue) 
        lst[index] = 1      # That index will be 1. Eg: [0,0,1,0, ... ,0]
        a = a + lst
        
        teams = ['Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders',
                'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
        
        # Batting Team
        batting_team = request.form['batting-team']
        lst = [0] * 8
        index = teams.index(batting_team)
        lst[index] = 1
        a = a + lst

        # Bowling Team
        bowling_team = request.form['bowling-team']
        lst = [0] * 8
        index = teams.index(bowling_team)
        lst[index] = 1
        a = a + lst

        if batting_team==bowling_team and batting_team!='none' and bowling_team!='none':
            return render_template('home.html',val='Batting team and Bowling team can\'t be the same and none of the fields can be empty.')



        overs = request.form['overs']
        runs = request.form['runs']
        wickets = request.form['wickets']
        runs_in_prev_5 = request.form['runs_in_prev_5']
        wickets_in_prev_5 = request.form['wickets_in_prev_5']

        if overs=='' or runs=='' or wickets=='' or runs_in_prev_5=='' or wickets_in_prev_5=='':
            return render_template('home.html',val='None of the fields can be empty!')

        overs = float(overs)
        runs = int(runs)
        wickets = int(wickets)
        runs_in_prev_5 = int(runs_in_prev_5)
        wickets_in_prev_5 = int(wickets_in_prev_5)

        a = np.array(a).reshape(1,-1)

        b = [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
        b = np.array(b).reshape(1,-1)
        b = scaler.transform(b)

        data = np.concatenate((a,b),axis=1)


        my_prediction = int(regressor.predict(data)[0])
        print(my_prediction)

        return render_template('home.html', val=f'The final score will be around {my_prediction-5} to {my_prediction+5}.')


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
