import os
import torch
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# Define a more flexible BiLSTM model architecture that can adapt to the saved model
class BiLSTM(nn.Module):
    def __init__(self, input_size=None, hidden_size=128, num_layers=1, dropout=0.3, output_size=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer (bidirectional means *2 for hidden size)
        self.fc = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, output_size)

    def forward(self, x, arimax_pred=None):
        # x shape: (batch_size, seq_length, input_size)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output from the last time step
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)

        # If ARIMAX predictions are provided, blend them with LSTM predictions
        if arimax_pred is not None:
            # Convert arimax_pred to tensor if it's not already
            if not isinstance(arimax_pred, torch.Tensor):
                arimax_pred = torch.tensor(arimax_pred, dtype=torch.float32).to(x.device)

            # Ensure arimax_pred has the right shape (batch_size, 1)
            if len(arimax_pred.shape) == 1:
                arimax_pred = arimax_pred.unsqueeze(1)

            # Adjust ARIMAX prediction length to match LSTM output length
            arimax_pred = arimax_pred[-out.shape[0]:]

            # Hybrid prediction (simple average for now)
            out = 0.5 * out + 0.5 * arimax_pred

        return out

# Load the most recent model
def get_most_recent_model_timestamp(folder='models'):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Model folder '{folder}' not found.")
    
    model_files = [f for f in os.listdir(folder) if f.startswith('model_info_')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in '{folder}'.")
    
    # Extract timestamps and find the most recent one
    timestamps = [f.replace('model_info_', '').replace('.pkl', '') for f in model_files]
    return max(timestamps)

class BitcoinPredictor:
    def __init__(self, model_folder='models'):
        self.model_folder = model_folder
        self.timestamp = get_most_recent_model_timestamp(model_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Loading model with timestamp: {self.timestamp}")
        self.load_model()

    def load_model(self):
        # Load model info
        model_info_path = os.path.join(self.model_folder, f'model_info_{self.timestamp}.pkl')
        print("\n\nLoading model info from:", model_info_path)
        model_info = joblib.load(model_info_path)
        self.sequence_length = model_info['sequence_length']

        # Load BiLSTM model
        bilstm_path = os.path.join(self.model_folder, f'bilstm_model_{self.timestamp}.pth')
        model_state = torch.load(bilstm_path, map_location=torch.device('cpu'))
        
        # Determine if the model has one or two LSTM layers
        # num_layers = 1  # Default to 1 layer
        # for key in model_state.keys():
        #     if "lstm.weight_ih_l1" in key:
        #         num_layers = 2
        #         break
                
        # # Get input size from the first layer's weight matrix
        # input_size = None
        # if "lstm.weight_ih_l0" in model_state:
        #     input_size = model_state["lstm.weight_ih_l0"].shape[1]
        
        # # Get hidden size from the first layer's weight matrix
        # hidden_size = None
        # if "lstm.weight_hh_l0" in model_state:
        #     hidden_size = model_state["lstm.weight_hh_l0"].shape[0] // 2  # Divide by 2 because it's bidirectional
        
        # if input_size is None or hidden_size is None:
        #     raise ValueError("Could not determine model architecture from saved state")
            
        # Infer architecture from state_dict
        input_size = model_state["lstm.weight_ih_l0"].shape[1]
        hidden_size = model_state["lstm.weight_hh_l0"].shape[1]
        num_layers = 1 if "lstm.weight_ih_l1" not in model_state else 2

        print(f"Detected model architecture - input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}")
        
        # Create model with correct architecture
        self.bilstm_model = BiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Load state dict
        try:
            self.bilstm_model.load_state_dict(model_state)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try to load with strict=False
            self.bilstm_model.load_state_dict(model_state, strict=False)
            print("Model loaded with strict=False")
            
        self.bilstm_model.to(self.device)
        self.bilstm_model.eval()

        # Load ARIMAX model
        arimax_path = os.path.join(self.model_folder, f'arimax_model_{self.timestamp}.pkl')
        self.arimax_model = joblib.load(arimax_path)

        # Load scalers
        self.feature_scaler = joblib.load(os.path.join(self.model_folder, f'feature_scaler_{self.timestamp}.pkl'))
        self.target_scaler = joblib.load(os.path.join(self.model_folder, f'target_scaler_{self.timestamp}.pkl'))

    def generate_features(self, current_price, volume, market_cap, polarity, subjectivity, days_to_predict):
        """Generate features for prediction based on user inputs and historical data patterns"""
        # Create a DataFrame with the initial values
        today = datetime.now()
        print("\n\ntoday:", today)
        dates = [today + timedelta(days=i) for i in range(self.sequence_length + days_to_predict)]
        
        df = pd.DataFrame({
            'date': dates,
            'price': [current_price] * (self.sequence_length + days_to_predict),
            'total_volume': [volume] * (self.sequence_length + days_to_predict),
            'market_cap': [market_cap] * (self.sequence_length + days_to_predict),
            'polarity': [polarity] * (self.sequence_length + days_to_predict),
            'subjectivity': [subjectivity] * (self.sequence_length + days_to_predict)
        })
        
        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Create additional features
        df['price_rolling_mean_7d'] = df['price'].rolling(window=7).mean().fillna(method='bfill')
        df['volume_rolling_mean_7d'] = df['total_volume'].rolling(window=7).mean().fillna(method='bfill')
        df['price_lag_1'] = df['price'].shift(1).fillna(method='bfill')
        df['price_lag_7'] = df['price'].shift(7).fillna(method='bfill')
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Add target column (will be predicted)
        df['target'] = df['price'].shift(-1).fillna(method='ffill')
        
        return df

    def prepare_data_for_prediction(self, df):
        """Prepare data for both ARIMAX and BiLSTM models"""
        try:
            # ARIMAX features
            y_arimax = df['price']
            X_arimax = df[['total_volume', 'market_cap', 'polarity', 'subjectivity',
                           'price_rolling_mean_7d', 'volume_rolling_mean_7d',
                           'day_of_week', 'month']]
            
            # BiLSTM features - all columns except target
            features = df.drop('target', axis=1).values
            
            # Check if feature scaler is properly fitted
            if not hasattr(self.feature_scaler, 'n_features_in_'):
                print("Warning: Feature scaler not properly fitted")
                # Use a simple min-max scaling as fallback
                min_vals = features.min(axis=0)
                max_vals = features.max(axis=0)
                scaled_features = (features - min_vals) / (max_vals - min_vals + 1e-10)
            else:
                # Use the loaded scaler
                try:
                    scaled_features = self.feature_scaler.transform(features)
                except Exception as e:
                    print(f"Error scaling features: {e}")
                    # If there's a mismatch in the number of features, we need to adapt
                    if hasattr(self.feature_scaler, 'feature_names_in_'):
                        print(f"Expected features: {self.feature_scaler.feature_names_in_}")
                    # Use simple scaling as fallback
                    min_vals = features.min(axis=0)
                    max_vals = features.max(axis=0)
                    scaled_features = (features - min_vals) / (max_vals - min_vals + 1e-10)
            
            # Create sequences for BiLSTM
            X_lstm = []
            for i in range(len(scaled_features) - self.sequence_length + 1):
                X_lstm.append(scaled_features[i:i + self.sequence_length])
            
            X_lstm = np.array(X_lstm)
            
            return X_lstm, X_arimax
            
        except Exception as e:
            print(f"Error in preparing data: {e}")
            raise

    def predict_future_prices(self, current_price, volume, market_cap, polarity, subjectivity, days_to_predict):
        """Predict future prices for the specified number of days"""
        try:
            # Generate features for the prediction period
            df = self.generate_features(current_price, volume, market_cap, polarity, subjectivity, days_to_predict)
            
            # Initial data preparation
            X_lstm, X_arimax = self.prepare_data_for_prediction(df)
            
            predictions = []
            dates = []
            
            # Make the initial prediction
            X_lstm_tensor = torch.FloatTensor(X_lstm[0:1]).to(self.device)

            print("\ndf: ",df)
            print("\ndf-info: ",df.info)
            
            # Make iterative predictions one day at a time
            for i in range(days_to_predict):
                # Get the current date
                prediction_date = df.index[self.sequence_length + i]
                print("\n\nPrediction date:", prediction_date)
                dates.append(prediction_date)
                
                # Get ARIMAX prediction for this time step
                try:
                    X_arimax.index = df.index

                    X_arimax_step = X_arimax.iloc[self.sequence_length + i:self.sequence_length + i + 1]
                    arimax_pred = self.arimax_model.forecast(steps=1, exog=X_arimax_step)
                    arimax_pred_value = arimax_pred.values[0] if hasattr(arimax_pred, 'values') else arimax_pred[0]
                    arimax_pred_scaled = self.target_scaler.transform([[arimax_pred_value]])
                except Exception as e:
                    print(f"ARIMAX prediction error: {e}")
                    arimax_pred_scaled = None
                
                # Get BiLSTM prediction
                with torch.no_grad():
                    if arimax_pred_scaled is not None:
                        lstm_pred = self.bilstm_model(X_lstm_tensor, 
                                                      arimax_pred=torch.FloatTensor(arimax_pred_scaled).to(self.device))
                    else:
                        lstm_pred = self.bilstm_model(X_lstm_tensor)
                
                # Convert prediction to numpy and rescale
                lstm_pred_np = lstm_pred.cpu().numpy()
                predicted_price = self.target_scaler.inverse_transform(lstm_pred_np)[0][0]
                predictions.append(predicted_price)
                
                # Update the price in the dataframe for the next prediction
                if i < days_to_predict - 1:
                    # Update the price for the next day
                    df.iloc[self.sequence_length + i + 1, df.columns.get_loc('price')] = predicted_price
                    
                    # Recalculate rolling features for the updated price
                    df['price_rolling_mean_7d'] = df['price'].rolling(window=7).mean().fillna(method='bfill')
                    df['price_lag_1'] = df['price'].shift(1).fillna(method='bfill')
                    df['price_lag_7'] = df['price'].shift(7).fillna(method='bfill')
                    
                    # Prepare data for the next prediction
                    X_lstm, X_arimax = self.prepare_data_for_prediction(df)
                    X_lstm_tensor = torch.FloatTensor(X_lstm[i+1:i+2]).to(self.device)
            
            return dates, predictions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise

    def plot_predictions(self, dates, predictions):
        """Plot the predicted prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions, marker='o', linestyle='-', color='blue')
        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot as a temporary file
        temp_file = "temp_prediction_plot.png"
        plt.savefig(temp_file)
        plt.close()
        
        return temp_file

# Gradio interface function with better error handling
def predict_bitcoin_price(current_price, volume, market_cap, polarity, subjectivity, days_to_predict):
    """Gradio function to predict Bitcoin price"""
    try:
        # Check inputs
        if current_price <= 0:
            return "Error: Current price must be positive", None
        if volume <= 0:
            return "Error: Volume must be positive", None
        if market_cap <= 0:
            return "Error: Market cap must be positive", None
        if days_to_predict < 1 or days_to_predict > 30:
            return "Error: Days to predict must be between 1 and 30", None
            
        # Create predictor and make predictions
        predictor = BitcoinPredictor()
        dates, predictions = predictor.predict_future_prices(
            float(current_price),
            float(volume),
            float(market_cap),
            float(polarity),
            float(subjectivity),
            int(days_to_predict)
        )
        
        # Format dates for better readability
        formatted_dates = [d.strftime("%Y-%m-%d") for d in dates]

        print("\nfromatted_dates:", formatted_dates)
        
        # Create a table of predictions
        prediction_data = {
            "Date": formatted_dates,
            "Predicted Price (USD)": [f"${p:.2f}" for p in predictions]
        }
        df_predictions = pd.DataFrame(prediction_data)
        
        # Plot the predictions
        plot_image = predictor.plot_predictions(dates, predictions)
        
        return df_predictions, plot_image
    
    except FileNotFoundError as e:
        return f"Model not found: {str(e)}", None
    except Exception as e:
        return f"Error: {str(e)}", None

# Create the Gradio interface
with gr.Blocks(title="Bitcoin Price Predictor 2025") as app:
    gr.Markdown("# Bitcoin Price Prediction App")
    gr.Markdown("Predict future Bitcoin prices using a hybrid ARIMAX-BiLSTM model")
    
    with gr.Row():
        with gr.Column():
            current_price = gr.Number(label="Current Bitcoin Price (USD)", value=92000)
            volume = gr.Number(label="Trading Volume (USD)", value=37000000000)
            market_cap = gr.Number(label="Market Cap (USD)", value=1820000000000)
            polarity = gr.Slider(label="Sentiment Polarity (-1 to 1)", minimum=-1, maximum=1, value=0.65, step=0.01)
            subjectivity = gr.Slider(label="Sentiment Subjectivity (0 to 1)", minimum=0, maximum=1, value=0.5, step=0.01)
            days = gr.Slider(label="Days to Predict", minimum=1, maximum=30, value=7, step=1)
            predict_btn = gr.Button("Predict Price", variant="primary")
        
        with gr.Column():
            output_table = gr.DataFrame(label="Price Predictions")
            output_plot = gr.Image(label="Price Trend")
    
    predict_btn.click(
        fn=predict_bitcoin_price,
        inputs=[current_price, volume, market_cap, polarity, subjectivity, days],
        outputs=[output_table, output_plot]
    )
    
    gr.Markdown("""
    ## How to use this app:
    
    1. Enter the current Bitcoin price in USD
    2. Provide the current trading volume and market cap
    3. Set the sentiment analysis values:
       - **Polarity**: Ranges from -1 (negative sentiment) to 1 (positive sentiment)
       - **Subjectivity**: Ranges from 0 (factual) to 1 (subjective/opinion)
    4. Select how many days into the future you want to predict
    5. Click "Predict Price" to see the forecast
    
    The prediction is based on a hybrid model combining ARIMAX time series analysis with Bidirectional LSTM neural networks.
    """)

if __name__ == "__main__":
    app.launch()