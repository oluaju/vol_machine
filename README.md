# Options Implied Volatility Prediction

![project2.png](/resources/images/project2.png)

Performance of a financial asset is usually measured via its return. The dispersion in the returns, as well as their distribution of outcomes, is described by the asset’s volatility. Volatility is an essential component of financial derivatives and has been extensively researched in the areas of finance, risk management and policy making. In this case study, we focused on two measures of volatilities: 
1. historical volatility (RV), which can be observed from the historical returns and realized at a certain point in time; 
2. implied volatility (IV), which is not directly observable, but output from the Black Scholes Pricing model that outputs IV as a proxy for the derivatives price, or what the market is implying the volatility will be. 

The goal of this project is to answer five overarching questions: 
1. Do the corelationships significantly impact the multi-step forecast of the IVS? 
2. Can TS Stat Models accurately predict multi-step forecast?
3. Does recurrent neural network architecture outperform traditional time series models in a multi-step out-of-sample forecast of the IVS?
4. Can the combination of RNN with CNN model drive better predictability
5. What we could better or to improve the performance of the model

In Summary, this project attempts to identify the best suited model and how hyperparameter tuning could play a role in predicting close to the realized Volatility n-days ahead.


Plan:

We will use Timeseries models such RMA, ARIMA and GARCH Models which are known to produce high accuracy forecasts for short look ahead periods
and Deep Learning models such as RNN and CNN for multi-step forecasting and compare the results. 

1. [Prepare the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Build and train Timeseries Models](#build-and-train-timeseries-model)
3. [Build and train custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
4. [Build and train CNNs](#build-and-train-CNN-model)
5. [Evaluate the performance of each model](#evaluate-the-performance-of-each-model)

- - -

### Files

[TimeSeries Model Notebook](/ibmvix_GARCH.ipynb)

[MultiVariate LSTM Notebook](/ibmvix.ipynb)

[MultiVariate CNN Notebook](/cnn_options_vol_predict.ipynb)

- - -

## Instructions

### Data Sources: 
1. ALPACA API for Daily close prices data for selected tickers [IBM, AAPL, GS & AMZN]
2. VIX data from Fred.stlouisfed.org

### Prepare the data for training and testing

1. Data Cleansing:
Data sourced from the publishers like ALPACA and Fred.Stlousfed is ready to be used without having to check for dataquality aspects. 

2. Data pre-processing:

* Slice the data using the window of time function for the n day window.
* For Timeseries models we will use the Vix Dataset 
* For multivariate forecasting models We will use previous closing prices to caluculate the Realized Volatility to predict Implied Volatility.
* We will use  70% of the data for training and 30% of the data for testing in each model
* using MinMaxScaler we will scale X and y values for the model. for Multivariate & multi dimensional data set, we will create scaler matrix of 
MinMaxScalers to suppoort Vectors of more than 2 dimensions. 
* In case of multistep forecasting, the dimensions of our samples will be more than 2 dimensions. in preparing hte LSTM models, the input shape has to be in the form of (n_samples, n_features and n_steps) which is a 3 dimensional vector. using Reshape method, x_train & x_test data sets are converted into the input shape for the LSTM models.


### Build and train TimeSeries Models

In the Jupyter Notebook ibmvix_GARCH.ipynb, we built ARMA, ARIMA and GARCH models including Multivariate forecasting using VAR (Vector Auto Regression) 

* In one notebook, we fit the data using the FNG values. In the second notebook, only closing prices are used.
* In order to compare the models, we will use the same parameters and training steps for each model. 



### Build and train custom LSTM RNNs

In ibmvix.ipynb Jupyter Notebook, We built custom LSTM RNN architectures for the following predictions
1. predicting IV using RV
2. predicting IV using RV, Volume
3. predicting IV using Technical Indicators by feature engineering. 


### Build and train CNNs

In cnn_options_vol_predict.ipynb Jupyter Notebook, We built CNN architectures for the following predictions
1. predict next days movement using Open, high, low & close features and normalizing them by calculating the Z-Scores
2. predict next 5 days movement - up or down.


#### Hyper Parameter Tuning
1. for each of the models, the hyper parameters such as the n_nodes, batch_size, epochs, Activators are tuned to find the optimal parameters for
training LSTM model. 

### Evaluate the performance of each model and answer the below questions:

1. Can TS stat models predict Implied voltility acurately over long horizons?

![TS_Stat_model_predictions.png](/resources/images/TS_Stat_model_predictions.PNG)
![VAR_5days.PNG](/resources/images/VAR_5days.PNG)

Observation: 
*From the above plotted results, TimeSeries Models do predict both direction and scale for short look ahead period (1 day) with reasonably high accuracy. When predicting multi-step forecast, TimeSeries models lose the ground on direction & scale. 

2. Does recurrent neural network architecture significantly outperform traditional time series models in a multi-step out-of-sample forecast of the IVS?

![uV_LSTM_predictions.png](/resources/images/uV_LSTM_predictions.PNG)

Observation: 
* From the above plotted results of Univariate nth day forecasting, RNN Recurrent Neural Network model predicts both direction and scale with reasonably high accuracy. in the above example, we predicted 5th day ahead using past 5 day window of data.


3. Do the cointegrated relationships significantly impact the multi-step forecast of the IVS? 

![MV_LSTM_predictions.png](/resources/images/MV_LSTM_predictions.PNG)

Observation: 
* When predicting using multiple features, the results vary depending on the corelationship of the features.  The model was testing using Volume and Realized volatility. the results are not overwhelming but have prompted to use additional features that have high corelation co-efficients. 

![FE_LSTM_predictions.png](/resources/images/FE_LSTM_predictions.PNG)

Observation: 
* When tried using technical indicators are additional features the results didnt improve much. 

* probably features such as Option put and call price, Option volume etc., would help predict mmore accurately compared to the technical indicators which may not directly influence the Options Implied volatility. Also the Stock Sentiment score could be another important feature which could influence the prediction outcomes. 

4. Can CNN model accurately predict the Implied ovalitility movement or direction ?
![MV_CNN_1_Day_predictions.png](/resources/images/MV_CNN_5_Day_predictions.PNG)

* CNN MModel Architecture
![cnn.png](/resources/images/cnn.png)


Observation: 
* We tried using CNN to predict next day's movement (up/down) with 98% accuracy. 

4. What can be improved or could be done better?
* Instead of using feature engineered data, we should get the features such as Options put and call prices, option volume etc., that are more tightly corelated to the Implied Volatility that could drive the accuracy of predictions. 
* also, predicting long horizons is not practical as it will be prone to errors due to the fact that there are many other parameters that will drive the volatility of options or stock prices which the model cannot predict. this project is a good example to demonstrate different model behaviours but the not necessarily to predict long horizons. 


- - -
