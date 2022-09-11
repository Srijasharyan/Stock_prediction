
# Stock Market Price Prediction(Deep Learning Project)

<img width="937" alt="image" src="https://user-images.githubusercontent.com/79760252/189542086-3ec5c872-de9f-4605-a487-08a0253a12ee.png">


### Deployed link-:(https://stockforecaster.herokuapp.com/)



### Structure of the Project Folder
- main.py - Consists of deep learning model for processing raw data from yahoo finance api
- app.py - Consists of callback functions and actual html/css models for building actual webpage layout
- requirements.txt - Consists of libraries required for importing dependensies 


### Methodology

- import the required stock data from yahoo finance API

- Data preprocessing which transforms data into training sets, testing sets, scaling  model and last price window sequence 

- Build deep learning model with user dynamic window sequence, neural layers, feature columns, loss and optimizer 

- Final algorithm to predict future stock price for variant number of days with recursive window stock sequence 




