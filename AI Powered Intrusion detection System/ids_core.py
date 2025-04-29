import pandas as pd
import numpy as np
import pickle
import pyshark
import threading
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, 
                            confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os

class IDS:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = {}
        self.feature_names = None
        self.alerts = []
        self.alert_lock = threading.Lock()
        self.capture = None
        self.capture_thread = None
        self.stop_capture = threading.Event()
        
    def load_data(self, train_path, test_path=None):
        """Load and preprocess NSL-KDD dataset"""
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'label', 'difficulty_level'
        ]
        
        print("Loading training data...")
        train_df = pd.read_csv(train_path, names=column_names)
        train_df.drop('difficulty_level', axis=1, inplace=True)
        
        if test_path:
            print("Loading test data...")
            test_df = pd.read_csv(test_path, names=column_names)
            test_df.drop('difficulty_level', axis=1, inplace=True)
        else:
            train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # Create binary labels (0 for normal, 1 for attack)
        train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Prepare features and labels
        X_train = train_df.drop(['label', 'binary_label'], axis=1)
        y_train = train_df['binary_label']
        X_test = test_df.drop(['label', 'binary_label'], axis=1)
        y_test = test_df['binary_label']
        
        # One-hot encode categorical features
        cat_cols = ['protocol_type', 'service', 'flag']
        for col in cat_cols:
            self.label_encoder[col] = LabelEncoder()
            X_train[col] = self.label_encoder[col].fit_transform(X_train[col])
            X_test[col] = self.label_encoder[col].transform(X_test[col])
        
        # Standardize numerical features
        numeric_cols = [col for col in X_train.columns if col not in cat_cols]
        self.scaler = StandardScaler()
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        self.feature_names = X_train.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train Random Forest classifier with hyperparameter tuning"""
        print("Training Random Forest model...")
        
        # Base model
        self.model = RandomForestClassifier(n_estimators=100, 
                                          random_state=42, 
                                          n_jobs=-1,
                                          class_weight='balanced')
        self.model.fit(X_train, y_train)
        
        # Hyperparameter tuning (optional)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2
        )
        
        print("Performing grid search...")
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('static/confusion_matrix.png')
        plt.close()
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importance (Top 20)')
        plt.barh(range(len(indices)), feature_importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig('static/feature_importance.png')
        plt.close()
    
    def save_model(self, path='models/random_forest_model.pkl'):
        """Save trained model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/random_forest_model.pkl'):
        """Load trained model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.feature_names = data['feature_names']
        print(f"Model loaded from {path}")
    
    def preprocess_packet(self, packet):
        """Preprocess a network packet for prediction using PyShark"""
        try:
            features = {
                'duration': 0,
                'protocol_type': packet.transport_layer if hasattr(packet, 'transport_layer') else 0,
                'service': packet[packet.transport_layer].dstport if hasattr(packet, 'transport_layer') else 0,
                'flag': 'SF',  # Simplified flag
                'src_bytes': int(packet.length) if hasattr(packet, 'length') else 0,
                'dst_bytes': 0,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': 1,
                'srv_count': 1,
                'serror_rate': 0,
                'srv_serror_rate': 0,
                'rerror_rate': 0,
                'srv_rerror_rate': 0,
                'same_srv_rate': 0,
                'diff_srv_rate': 0,
                'srv_diff_host_rate': 0,
                'dst_host_count': 1,
                'dst_host_srv_count': 1,
                'dst_host_same_srv_rate': 0,
                'dst_host_diff_srv_rate': 0,
                'dst_host_same_src_port_rate': 0,
                'dst_host_srv_diff_host_rate': 0,
                'dst_host_serror_rate': 0,
                'dst_host_srv_serror_rate': 0,
                'dst_host_rerror_rate': 0,
                'dst_host_srv_rerror_rate': 0
            }
            
            # Create DataFrame
            df = pd.DataFrame([features])
            
            # Encode categorical features
            for col in ['protocol_type', 'service', 'flag']:
                if col in self.label_encoder:
                    df[col] = self.label_encoder[col].transform([str(features[col])])[0]
            
            # Scale numerical features
            numeric_cols = [col for col in df.columns if col not in ['protocol_type', 'service', 'flag']]
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
            return df
        except Exception as e:
            print(f"Error processing packet: {e}")
            return None
    
    def packet_handler(self, packet):
        """Handle incoming packets"""
        df = self.preprocess_packet(packet)
        if df is not None and self.model is not None:
            prediction = self.model.predict(df)
            proba = self.model.predict_proba(df)[0][1]
            
            if prediction[0] == 1:  # Attack detected
                alert = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'src_ip': packet.ip.src if hasattr(packet, 'ip') else 'N/A',
                    'dst_ip': packet.ip.dst if hasattr(packet, 'ip') else 'N/A',
                    'src_port': packet[packet.transport_layer].srcport if hasattr(packet, 'transport_layer') else 'N/A',
                    'dst_port': packet[packet.transport_layer].dstport if hasattr(packet, 'transport_layer') else 'N/A',
                    'protocol': packet.transport_layer if hasattr(packet, 'transport_layer') else 'N/A',
                    'confidence': f"{proba:.2f}",
                    'type': 'Unknown Attack'
                }
                
                with self.alert_lock:
                    self.alerts.append(alert)
                
                print(f"ALERT: {alert}")
    
    def start_sniffing(self, interface='eth0'):
        """Start sniffing network traffic using PyShark"""
        def capture_loop():
            self.capture = pyshark.LiveCapture(
                interface=interface,
                display_filter='ip'  # Only capture IP packets
            )
            
            print(f"Starting packet capture on interface {interface}...")
            for packet in self.capture.sniff_continuously():
                if self.stop_capture.is_set():
                    break
                self.packet_handler(packet)
        
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop_sniffing(self):
        """Stop the packet capture"""
        if self.capture_thread and self.capture_thread.is_alive():
            self.stop_capture.set()
            if self.capture:
                self.capture.close()
            self.capture_thread.join(timeout=2)
            print("Packet capture stopped")
    
    def get_alerts(self):
        """Get current alerts"""
        with self.alert_lock:
            return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all alerts"""
        with self.alert_lock:
            self.alerts = []