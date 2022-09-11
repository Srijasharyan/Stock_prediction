
import time
from dash import Dash, html, dcc, Input, Output
from dash import ctx
import pandas as pd
import plotly.graph_objects as go
import datetime as date
from main import print_data,process_data,build_model,predict_future
import tensorflow as tf
import dash

app = Dash(__name__)
server=app.server


app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = html.Div(children=[

    html.H1(children='Stock price prediction', style={'color': 'gray', 'text-align': 'center'}),
    html.Div( children=[
        html.Div(children=[
            html.H4(children='Enter stock', style={'color': 'gray', 'text-align': 'left'}),
            dcc.Input(id='input-1-state', type='text',  style={'text-align': 'center'}, value='eg. MSFT'),
            html.Br(),
            html.H4(children='Select time interval', style={'color': 'gray', 'text-align': 'left'}),
            dcc.DatePickerRange(
            id='date-picker-range',
            start_date_placeholder_text= 'Start',
            end_date_placeholder_text='End',
        ),
            html.Br(),
            html.Br(),
            html.Button(id='submit-button', n_clicks=0, children='Submit'),
            html.Br(),
            html.Div(id='text-output',style={'color': 'gray', 'text-align': 'center','fontSize': 25}),
            html.Br(),
            dcc.Graph(id='graph-output',figure=go.Figure()),
        

            ]),
       



               
            html.H4(children='Enter no. of days for prediction', style={'color': 'gray', 'text-align': 'left'}),
            
            dcc.Input(id='input-2-state', type='text',  style={'text-align': 'left'}, value='eg. 2'),
            html.Br(),
            html.Br(),
            html.Button(id='predict-button', n_clicks=0, children='Predict'),
            dcc.Loading(
                id="loading-1",
                type="default",
                children=html.Div(dcc.Graph(id='graph-output2',figure=go.Figure()))
        ),
            
            
            dcc.Store(id='store',data=[],storage_type='memory')

])
   
        
])

# s2=[] 
# e2=[]
# input_text=""
@app.callback(
    Output(component_id='text-output', component_property='children'),
    Output(component_id='graph-output',component_property='figure'),
    Output('store','data'),
    # Output(component_id='graph-output2',component_property='figure'),
    Input(component_id='input-1-state', component_property='value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('submit-button', 'n_clicks'),
    # Input(component_id='input-2-state', component_property='value'),
    # Input('predict-button', 'n_clicks'),
    prevent_initial_call=False

)


def Output1(input_text,start_date,end_date,btn1):
    s=[]
    e=[]
    # fig2 = go.Figure()
    fig = go.Figure()
    # PreventUpdate prevents ALL outputs updating
    if start_date is None:
        raise dash.exceptions.PreventUpdate
    if end_date is None:
        raise dash.exceptions.PreventUpdate 
    if input_text is None:
        raise dash.exceptions.PreventUpdate 

    else:
        if start_date is not None and end_date is not None:
            s= start_date.split("-")
            e= end_date.split("-")
        data=print_data(ticker=input_text,start=date.datetime(int(s[0]),int(s[1]),int(s[2])), end=date.datetime(int(e[0]), int(e[1]), int(e[2])))
        fig = go.Figure([go.Scatter(x=data['date'], y=data['Close'])])
        array=[input_text,s,e]  
        if "submit-button" == ctx.triggered_id:
            return f"{input_text} stock price",fig, array    #,fig2
    return


@app.callback(
    

    Output(component_id="graph-output2", component_property="figure"),
    Input(component_id='input-2-state', component_property='value'),

    Input('predict-button', 'n_clicks'),
    Input('store', 'data'),
    prevent_initial_call=False
)
def Output2(input_int,btn2,data):
    fig2 = go.Figure()
    if input_int is  None:
        raise dash.exceptions.PreventUpdate
    else:
        if "predict-button" == ctx.triggered_id:
            
            Data=process_data(ticker=data[0], feature_columns=['Close'],start=date.datetime(int(data[1][0]),int(data[1][1]),int(data[1][2])),end=date.datetime(int(data[2][0]),int(data[2][1]),int(data[2][2])), 
                    n_steps=5, scale=True, shuffle=True, lookup_step=1, split_by_date=True,test_size=0)


            Model=build_model(sequence_length=5, n_features=1, units=80, cell=tf.keras.layers.LSTM, n_layers=3, dropout=0.2,
                loss="mean_absolute_error", optimizer="adam", bidirectional=True)

            yy=predict_future(Model,Data,int(input_int),25,32,1,5)
            xx=list(range(1,int(input_int)+1))
            fig2=go.Figure([go.Scatter(x=xx,y=yy)])
            time.sleep(1)

    return fig2

if __name__ == '__main__':
    app.run_server(debug=True)




