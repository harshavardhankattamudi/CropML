from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib
import pandas as pd
from flask_cors import CORS
import os
import json
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import numpy as np

# Adjust static and template folder paths for Vercel
app = Flask(__name__, static_folder='../static', template_folder='../templates')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# ... existing code from app.py ...

# At the end, expose the app as 'app' for Vercel
# (Vercel expects 'app' to be the Flask app instance) 