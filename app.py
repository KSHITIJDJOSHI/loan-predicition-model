from flask import Flask,render_template,request,url_for
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('randomforest.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict",methods=["POST","GET"])

def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    new_data=scaler.transform(np.array(data).reshape(1,-1))

    output=model.predict(new_data)[0]
    if output==1:
        return "congratulation you are eligible for loan "   
    return "sorry but your are not eligibable for loan"







if __name__=="__main__":
    app.run(debug=True)