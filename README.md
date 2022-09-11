
# Stock Market Price Prediction(Deep Learning Project)

<img width="937" alt="image" src="https://user-images.githubusercontent.com/79760252/189542375-e0d4a9a9-32f0-4774-a9de-5e6b21da439d.png"><img width="933" alt="image" src="https://user-images.githubusercontent.com/79760252/189542434-d7afa9a9-1f83-4d26-8c8d-b8dd3262eea7.png">



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




