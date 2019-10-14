import numpy as np
from encoder import Model
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="templates")

#https://modeldepot.io/afowler/sentiment-neuron
model = Model()

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Make prediction
        if request.form.get('text') != '':
            text_features = model.transform([request.form.get('text')])
            
            if round(np.asarray(text_features[:, 2388], dtype=np.float)[0]) == 0:
                sentiment=2
            elif round(np.asarray(text_features[:, 2388], dtype=np.float)[0]) > 0:
                sentiment=1
            else:
                sentiment=3

            return render_template('index.html', sentiment=sentiment)
            
    return render_template('index.html', sentiment='')
    
if __name__ == '__main__':
    app.run()