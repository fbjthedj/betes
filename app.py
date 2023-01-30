# # We will import our libraries 
# # ---
# # Numpy for scientific computations
# # Flask - as our web framework which will provide the following web components 
# # -> Flask           - core flask web components
# # -> request         - for information retrieval from our html file
# # -> jsonify         - for storing information in JSON Data. JSON Data is a format for passing data in the web.
# # -> render_template - for showing/rendering html files
# # -> Pickle          - for saving (serializing) and loading (de-serializing) our model
# # ---
# #
# #import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # We now start using flask by creating an instance. This of this like what we did 
# # when we create an instance of a model i.e. random_forest_classifier = RandomForestClassifier() 
# # but now for Flask()
# app = Flask(__name__)

# # We then load our random_forest_model using the pickle library
# # ---
# # We use 'rb' because the type of file of our model is a binary type.
# # For ease of understanding we din't want to the details pickle files.
# # ---
# # 
# model = pickle.load(open('random_forest_model.pkl', 'rb'))

# # We then start now creating our web application by starting with the route '/'
# # Our home function will return the index.html file that we created earlier.
# # ---
# #
# @app.route('/')
# def home():
#     return render_template('index.html')

# # The 
# # ---
# # In web applications, you can create routes of different nature i.e. GET, POST, UPDATE, etc.
# # The GET method usually displays content in a webpage. 
# # A route uses the GET method if not defined just as in the previous '/' method.
# # The POST method method is used to send and retrieve data from our web page to our server - vice versa.
# # In our case, if you have a look at the index.html form above, you will see that we are using the post method
# # as well as specifying the url/route i.e. predict that we want our form upon submission should act on.
# # <form action="{{ url_for('predict')}}" method="post">
# # ---
# # Let's quickly go through the following code that is executed once a user submits the form by clicking
# # the predict button. 
# #
# @app.route('/predict', methods = ['POST'])
# def predict():
    
#     # We get our input data (now features) from the index.html form.
#     # The method used in our case if one that iterates over the input fields in the form
#     int_features = [int(x) for x in request.form.values()]

#     # Then convert those features into a numpy array that our model understands
#     # During this step we could even perform feature engineering techniques
#     # to make sure our data is optimal for the model.
#     final_features = [np.array(int_features)]

#     # Then make our prediction for those features
#     prediction = model.predict(final_features)

#     # Round the predicted values to 2dp
#     outcome = round(prediction[0], 2)

#     # And return our predicted values to our index.html webpage, replacing the variable 
#     # prediction_text rendered in our page with our outcome.
#     # return render_template('index.html', prediction_text = outcome)

#     #output = round(prediction[0], 2)

#     return render_template('index.html', prediction_text='Your diabetes risk is $ {}'.format(outcome))

# # For now we are using the above two routes of '/' and '/predict' to achieve a simple web application
# # for a model solution that we deploy. If we wanted to have this web application to be used by other 
# # web applications or even mobile applications, we can create the following route with a POST method
# # to get input from a user and then later pass it to a 
# # This sort of rount would be what we would refer to as a Restful Application Programming Interface (Restful API).
# # A restful API would take input data and give output data in the form of an JSON format. 
# # An example of JSON data looks like this: person = '{"name":"John", "age":31, "city":"New York"}'.
# # It has some similarites with python dictionaries but different. JSON format can be thought 
# # of as a python dictionary that can be nested any number of times and passed over the web.
# # ---
# #
# @app.route('/predict_api', methods=['POST'])
# def predict_api():
    
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#     output = prediction[0]

#     return jsonify(output)



# # Then we execute all the above code in our server
# if __name__ == '__main__':
#     app.run(debug=True)          ## Running the app as debug==True


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('random_forest_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    
    
    features = [float(x) for x in request.form.values()]

   
    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    outcome = round(prediction[0], 2)
  
    return render_template('index.html', prediction_text='Your diabetes risk is assessed as {}'.format(outcome))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)          ## Running the app as debug==True