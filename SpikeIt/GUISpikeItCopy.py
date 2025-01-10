# Import necessary libraries 
import sys
import os
import time
import numpy as np
import pandas as pd
import serial  # For serial communication
import tensorflow as tf
from joblib import load  # Use joblib for loading scaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt  # For threading and GUI signals
from PyQt5.QtGui import QPixmap, QFont  # For image and font handling
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt  # For plotting data

# Paths to the trained model and scaler
model_path = "C:/Users/marti/Documents/POLIMI/SEMESTER 3/CSI/Hackathon-main/SpikesSantiago/MLPModel.h5"
scaler_path = "C:/Users/marti/Documents/POLIMI/SEMESTER 3/CSI/Hackathon-main/SpikesSantiago/scaler.pkl"

# Load the model and scaler
model = load_model(model_path)
scaler = load(scaler_path)

# Invalid phrases to ignore during preprocessing
invalid_phrases = [
    "Failed to read IMU data!",
    "Disconnected from NiclaIMU!",
    "Found NiclaIMU. Connecting...",
    "Connected to NiclaIMU!",
    "Service not found!",
    "IMU characteristic found!",
    "Received IMU Data:"
]

# Thread class for handling serial communication with an external device
class SerialThread(QThread):
    # Signal to send data to the GUI
    data_received = pyqtSignal(dict)

    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = False

    # Thread's main loop to handle serial communication
    def run(self):
        try:
            # Open serial port
            with serial.Serial(self.port, self.baud_rate, timeout=1) as ser:
                print("Connected to Portenta H7")
                ser.write(b"Hello, Portenta!\n")  # Send initial message
                time.sleep(0.1)
                self.running = True

                # Continuously read data from the serial port
                while self.running:
                    if ser.in_waiting > 0:  # Check if data is available
                        response = ser.readline().decode('utf-8').strip()
                        if response.startswith("Received IMU Data:"):
                            continue  # Skip specific messages

                        try:
                            # Parse received IMU data into a dictionary
                            imu_data = {
                                key: float(value)
                                for key, value in (pair.split(":") for pair in response.split())
                            }
                            self.data_received.emit(imu_data)  # Send data to the GUI
                        except ValueError:
                            print(f"Could not parse response: {response}")
        except serial.SerialException as e:
            print(f"Error: {e}")

    # Method to stop the thread
    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spike-It: Volleyball Telemetry")
        self.setGeometry(100, 100, 1920, 1080)  # Set window size to 1920x1080

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Background image
        self.background_label = QLabel(self)
        pixmap = QPixmap("C:/Users/marti/Documents/POLIMI/SEMESTER 3/CSI/Hackathon-main/palla.jpg")
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.background_label.setPixmap(scaled_pixmap)
        self.background_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.background_label)

        # Create a QLabel for the text
        label = QLabel("Spike-It: Volleyball Telemetry")
        label.setAlignment(Qt.AlignCenter)  # Align the text to the center

        # Set font and size
        font = QFont("Arial", 40)  # Font family and size
        font.setBold(True)         # Make the font bold (optional)
        label.setFont(font)

        # Add the label to the layout
        main_layout.addWidget(label)

        # Layout for buttons
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        # Button to browse file
        self.browse_button = QPushButton("Browse File")
        self.browse_button.setFixedSize(300, 60)
        self.browse_button.clicked.connect(self.browse_file)
        button_layout.addWidget(self.browse_button)

        # Button to start live data
        self.live_data_button = QPushButton("Live Data")
        self.live_data_button.setFixedSize(300, 60)
        self.live_data_button.clicked.connect(self.open_live_data_window)
        button_layout.addWidget(self.live_data_button)

        # Button to open training options
        self.training_button = QPushButton("Training (Under Dev.)")
        self.training_button.setFixedSize(300, 60)
        self.training_button.clicked.connect(self.open_training_window)
        button_layout.addWidget(self.training_button)

        # Add buttons layout to the main layout
        main_layout.addLayout(button_layout)

        # Label for feedback
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: white;")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

    # Method to open the training window
    def open_training_window(self):
        """Open the TrainingWindow."""
        self.training_window = TrainingWindow()
        self.training_window.show()

    # Method to open a file dialog to browse for a file
    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select IMU Data File", "", "Text Files (*.txt);;All Files (*)", options=options
        )

        if file_path:
            self.result_label.setText(f"File selected: {os.path.basename(file_path)}")
            self.result_label.setStyleSheet("color: green; font-size: 16px;")  # Set text color to green and font size
            try:
                self.df = self.read_txt_file(file_path)  # Parse file into DataFrame
                self.open_results_window()  # Open results window
                self.open_selection_window()  # Open selection window
            except ValueError as e:
                self.result_label.setText(str(e))
                self.result_label.setStyleSheet("color: red; font-size: 16px;")
        else:
            self.result_label.setText("No file selected.")
            self.result_label.setStyleSheet("color: gray; font-size: 16px;")  # Set text color to gray for no selection
            
    # Method to parse the contents of the selected file
    def read_txt_file(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("Received IMU Data:"):
                    continue  # Skip specific lines
                line_data = {}
                for pair in line.strip().split():
                    if ":" in pair:
                        key, value = pair.split(":")
                        line_data[key.strip()] = float(value.strip())
                if line_data:
                    data.append(line_data)

        if not data:
            raise ValueError("The file is empty or does not contain valid data.")

        # Convert parsed data into a DataFrame
        df = pd.DataFrame(data)
        required_columns = {"QW", "QX", "QY", "QZ", "AX", "AY", "AZ", "GX", "GY", "GZ"}
        if not required_columns.issubset(df.columns):
            raise ValueError("The file must contain all required IMU keys.")
        return df

    # Method to open the results window
    def open_results_window(self):
        if hasattr(self, 'df'):
            self.results_window = ResultsWindow(self.df)
            self.results_window.show()

    # Method to open the selection window
    def open_selection_window(self):
        """Open the SelectionWindow."""
        if hasattr(self, 'df'):
            self.selection_window = SelectionWindow(self.df)
            self.selection_window.show()

    # Method to open the live data window
    def open_live_data_window(self):
        self.live_data_window = LiveDataWindow()
        self.live_data_window.show()

# Window to Open the training GUI
class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Options")
        self.setGeometry(400, 300, 400, 300)

        # Create a layout
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Add "Live" button
        self.live_button = QPushButton("Live")
        self.live_button.setFixedSize(200, 60)
        self.live_button.clicked.connect(self.handle_live_training)
        layout.addWidget(self.live_button)

        # Add "File" button
        self.file_button = QPushButton("File Selection")
        self.file_button.setFixedSize(200, 60)
        self.file_button.clicked.connect(self.handle_file_training)
        layout.addWidget(self.file_button)
    
    def handle_live_training(self):
        # Placeholder for handling live training
        print("Live Training selected.")

    def handle_file_training(self):
        """Open the FileTraining window."""
        self.file_training_window = FileTraining()
        self.file_training_window.show()

# Window to open File Training
class FileTraining(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Training")
        self.setGeometry(400, 300, 400, 300)

        # Create a layout
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Add "Select File" button
        self.select_file_button = QPushButton("Select File")
        self.select_file_button.setFixedSize(200, 60)
        self.select_file_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_file_button)

        # Add label for prediction results
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.result_label)

    def select_file(self):
        """Handle file selection and prediction."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select IMU Data File", "", "Text Files (*.txt);;All Files (*)", options=options
        )

        if file_path:
            # Preprocess and predict
            processed_data = self.preprocess_file(file_path)
            prediction = model.predict(processed_data)

            # Interpret the prediction
            confidence = prediction[0][0]
            if confidence >= 0.7:
                result = "Good Spike"
            else:
                result = "Bad Spike"

            # Display the prediction
            self.result_label.setText(f"Prediction: {result}\nConfidence: {confidence:.2f}")
        else:
            self.result_label.setText("No file selected.")

    def preprocess_file(self, file_path):
        """Preprocess the selected .txt file."""
        quaternion_data = []

        with open(file_path, 'r') as f:
            for line in f:
                # Treat invalid phrases as zero values
                if any(phrase in line for phrase in invalid_phrases):
                    quaternion_data.append([0, 0, 0, 0])  # QW, QX, QY, QZ
                    continue

                # Extract values
                try:
                    data_values = [float(pair.split(":")[1]) for pair in line.strip().split()]
                    quaternion_data.append(data_values[0:4])  # QW, QX, QY, QZ
                except (ValueError, IndexError):
                    quaternion_data.append([0, 0, 0, 0])  # Handle unexpected lines

        # Convert to NumPy array
        quaternion_data = np.array(quaternion_data)

        # Normalize using the loaded scaler
        quaternion_data = scaler.transform(quaternion_data)

        # Pad or truncate the data to match the model's input shape
        max_length = model.input_shape[1] // 4  # Each feature vector has 4 components
        quaternion_data = pad_sequences([quaternion_data], maxlen=max_length, padding="post", dtype="float32")

        # Flatten the data to match the MLP model input
        features = quaternion_data[0].flatten()

        return np.expand_dims(features, axis=0)  # Add batch dimension

# Window to display results after reading data from a file
class ResultsWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Results Window")
        self.setGeometry(150, 150, 1920, 1080)  # Set window size to 1920x1080
        self.df = df

        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Add tables for Quaternions, Accelerations (with magnitude), and Gyroscope (with magnitude)
        self.add_data_table(layout, self.df[['QW', 'QX', 'QY', 'QZ']], "Quaternions (QW, QX, QY, QZ)")
        self.add_data_table(layout, self.df[['AX', 'AY', 'AZ']], "Accelerations (AX, AY, AZ)", calculate_magnitude=True, magnitude_column_name="Acceleration Magnitude")
        self.add_data_table(layout, self.df[['GX', 'GY', 'GZ']], "Gyroscope (GX, GY, GZ)", calculate_magnitude=True, magnitude_column_name="Gyroscope Magnitude")

    def add_data_table(self, layout, data, title, calculate_magnitude=False, magnitude_column_name=None):
        """Helper method to add a table for specific data."""
        # Create a QLabel for the title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 25px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        # If magnitudes are to be calculated, compute them
        if calculate_magnitude and magnitude_column_name:
            data[magnitude_column_name] = np.sqrt((data**2).sum(axis=1))  # Compute row-wise magnitudes
            columns = data.columns.tolist()
        else:
            columns = data.columns.tolist()

        # Create a QTableWidget for the data
        table = QTableWidget(self)
        table.setRowCount(len(data))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)

        # Populate the table with data
        for row_idx, row in enumerate(data.itertuples(index=False)):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QTableWidgetItem(f"{value:.3f}"))

        # Adjust table appearance
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make the table read-only
        table.setStyleSheet("font-size: 22px; margin-bottom: 5px;")

        # Add the table to the layout
        layout.addWidget(table)

# Window to display options for data visualization
class SelectionWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Select an Option")
        self.setGeometry(200, 200, 600, 400)  # Set window size
        self.df = df

        # Create a layout for the selection buttons
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Add "Graphs" button
        self.graphs_button = QPushButton("Graphs")
        self.graphs_button.setFixedSize(200, 60)
        self.graphs_button.clicked.connect(self.open_graphs_window)
        layout.addWidget(self.graphs_button)

        # Add "Statistics" button
        self.statistics_button = QPushButton("Statistics")
        self.statistics_button.setFixedSize(200, 60)
        self.statistics_button.clicked.connect(self.open_statistics_window)
        layout.addWidget(self.statistics_button)

    def open_graphs_window(self):
        """Open the Graphs window."""
        self.graphs_window = GraphsWindow(self.df)
        self.graphs_window.show()

    def open_statistics_window(self):
        """Open the Statistics window."""
        self.statistics_window = StatisticsWindow(self.df)
        self.statistics_window.show()

# Window to display graphs of the data
class GraphsWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Graphs")
        self.setGeometry(300, 300, 1920, 1080)
        self.df = df

        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create matplotlib figure and canvas
        self.fig, self.axs = plt.subplots(3, 1, figsize=(20, 26))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Plot data
        self.plot_data()

    def plot_data(self):
        """Plot data from the dataframe."""
        # Quaternions
        self.axs[0].plot(self.df.index, self.df["QW"], label="QW")
        self.axs[0].plot(self.df.index, self.df["QX"], label="QX")
        self.axs[0].plot(self.df.index, self.df["QY"], label="QY")
        self.axs[0].plot(self.df.index, self.df["QZ"], label="QZ")
        self.axs[0].set_title("Quaternions")
        self.axs[0].legend()

        # Accelerations
        self.axs[1].plot(self.df.index, self.df["AX"], label="AX")
        self.axs[1].plot(self.df.index, self.df["AY"], label="AY")
        self.axs[1].plot(self.df.index, self.df["AZ"], label="AZ")
        self.axs[1].set_title("Accelerations m/s^2")
        self.axs[1].legend()

        # Gyroscope
        self.axs[2].plot(self.df.index, self.df["GX"], label="GX")
        self.axs[2].plot(self.df.index, self.df["GY"], label="GY")
        self.axs[2].plot(self.df.index, self.df["GZ"], label="GZ")
        self.axs[2].set_title("Gyroscope deg/s")
        self.axs[2].legend()

        # Refresh canvas
        self.canvas.draw()

# Window to display statistics of the data
class StatisticsWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Statistics")
        self.setGeometry(200, 200, 800, 600)
        self.df = df

        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Calculate magnitudes for accelerations and gyroscope data
        self.df["Acceleration Magnitude"] = np.sqrt(
            self.df["AX"]**2 + self.df["AY"]**2 + self.df["AZ"]**2
        )
        self.df["Gyroscope Magnitude"] = np.sqrt(
            self.df["GX"]**2 + self.df["GY"]**2 + self.df["GZ"]**2
        )

        # Calculate descriptive statistics
        stats = self.df.describe().T  # Transpose for better readability

        # Create a table to display statistics
        stats_table = QTableWidget(self)
    
        # Set table dimensions
        stats_table.setRowCount(len(stats))
        stats_table.setColumnCount(len(stats.columns) + 1)  # Add column for row headers
        stats_table.setHorizontalHeaderLabels(["Metric"] + list(stats.columns))
        stats_table.verticalHeader().setVisible(False)

        # Populate the table with statistics
        for row_idx, (index, row) in enumerate(stats.iterrows()):
            stats_table.setItem(row_idx, 0, QTableWidgetItem(index))
            for col_idx, value in enumerate(row):
                stats_table.setItem(row_idx, col_idx + 1, QTableWidgetItem(f"{value:.3f}"))

        # Adjust table appearance
        stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_table.setStyleSheet("font-size: 24px;")
        stats_table.setAlternatingRowColors(True)
        stats_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make the table read-only

        layout.addWidget(stats_table)

# Window to display live data using matplotlib plots
class LiveDataWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Data")
        self.resize(1920, 1080)  # Set window size to 1920x1080

        # Set up serial thread for live data
        self.serial_thread = SerialThread("COM17", 115200)
        self.serial_thread.data_received.connect(self.update_plots)

        # Set up matplotlib figure and canvas
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 18))  # Adjusted for HD resolution
        self.canvas = FigureCanvas(self.fig)

        # Layout for live data
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        # Add Start, Stop, Start Logging, and Stop Logging buttons
        self.start_button = QPushButton("Start Live Feed")
        self.stop_button = QPushButton("Stop Live Feed")
        self.start_logging_button = QPushButton("Start Logging")
        self.stop_logging_button = QPushButton("Stop Logging")

        # Set button sizes
        self.start_button.setFixedSize(180, 50)
        self.stop_button.setFixedSize(180, 50)
        self.start_logging_button.setFixedSize(180, 50)
        self.stop_logging_button.setFixedSize(180, 50)

        # Connect button signals
        self.start_button.clicked.connect(self.start_live_feed)
        self.stop_button.clicked.connect(self.stop_live_feed)
        self.start_logging_button.clicked.connect(self.start_logging)
        self.stop_logging_button.clicked.connect(self.stop_logging)

        # Add buttons to layout
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.start_logging_button)
        button_layout.addWidget(self.stop_logging_button)
        layout.addLayout(button_layout)

        # Central widget setup
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Buffers to store data for plotting
        self.timestamps = []
        self.quaternion_data = {"QW": [], "QX": [], "QY": [], "QZ": []}
        self.acceleration_data = {"AX": [], "AY": [], "AZ": []}
        self.gyroscope_data = {"GX": [], "GY": [], "GZ": []}
        self.logging_active = False
        self.logged_data = []
        
        # Flag to track the live feed state
        self.live_feed_active = False

    def start_live_feed(self):
        """Start the live data feed."""
        if not self.live_feed_active:
            self.live_feed_active = True
            self.serial_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stop_live_feed(self):
        """Stop the live data feed."""
        if self.live_feed_active:
            self.live_feed_active = False
            self.serial_thread.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def start_logging(self):
        """Start logging data."""
        self.logging_active = True
        self.logged_data = []  # Reset logged data
        self.start_logging_button.setEnabled(False)
        self.stop_logging_button.setEnabled(True)

    def stop_logging(self):
        """Stop logging data and prompt user to save."""
        self.logging_active = False
        self.start_logging_button.setEnabled(True)
        self.stop_logging_button.setEnabled(False)

        if self.logged_data:
            # Prompt user to save the logged data
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Logged Data", "", "Text Files (*.txt);;All Files (*)", options=options
            )
            if file_path:
                # Save logged data to a text file in a specific format
                with open(file_path, "w") as file:
                    for entry in self.logged_data:
                        imu_data_str = "Received IMU Data:\n" + " ".join(
                            f"{key}:{value:.3f}" for key, value in entry.items() if key != "timestamp"
                        )
                        file.write(imu_data_str + "\n")
                print(f"Data logged to: {file_path}")

    def update_plots(self, imu_data):
        """Update plots with new data received from the serial thread."""
        if not self.live_feed_active:
            return  # Skip updates if the feed is stopped

        current_time = time.time()
        self.timestamps.append(current_time)

        # Append data to respective buffers
        for key in self.quaternion_data.keys():
            self.quaternion_data[key].append(imu_data.get(key, 0))
        for key in self.acceleration_data.keys():
            self.acceleration_data[key].append(imu_data.get(key, 0))
        for key in self.gyroscope_data.keys():
            self.gyroscope_data[key].append(imu_data.get(key, 0))

        # Log data if logging is active
        if self.logging_active:
            imu_data["timestamp"] = current_time
            self.logged_data.append(imu_data)

        # Limit buffer size to the last 100 points
        if len(self.timestamps) > 100:
            self.timestamps.pop(0)
            for data in (self.quaternion_data, self.acceleration_data, self.gyroscope_data):
                for key in data.keys():
                    data[key].pop(0)

       # Update plot data with individual y-axis limits
        y_limits = [
            (-2, 2),          # Quaternion range
            (-150, 150),      # Acceleration range
            (-20000, 20000)   # Gyroscope range
        ]
        titles = ["Quaternion", "Acceleration m/s", "Gyroscope deg/s"]
        labels = [
            ["QW", "QX", "QY", "QZ"],
            ["AX", "AY", "AZ"],
            ["GX", "GY", "GZ"]
        ]
        data_sets = [self.quaternion_data, self.acceleration_data, self.gyroscope_data]

        for i, (data, y_lim, title, label_set) in enumerate(zip(data_sets, y_limits, titles, labels)):
            self.axs[i].cla()  # Clear the axis
            self.axs[i].set_title(title)
            self.axs[i].set_xlim(max(0, self.timestamps[-1] - 5), self.timestamps[-1])  # Rolling time window
            self.axs[i].set_ylim(*y_lim)  # Apply individual y-axis limits
            for key, label in zip(data.keys(), label_set):
                self.axs[i].plot(self.timestamps, data[key], label=label)
            self.axs[i].legend()

        # Redraw the canvas after all updates
        self.canvas.draw()

# Entry point of the program
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())