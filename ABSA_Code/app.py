import re
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from transformers import TFRobertaModel, AutoTokenizer, AdamWeightDecay
from preprocess import text_preprocess
import json
import os


tf.keras.utils.get_custom_objects().update({'TFRobertaModel': TFRobertaModel})
tf.keras.utils.get_custom_objects().update({'AdamWeightDecay': AdamWeightDecay})
model = tf.keras.models.load_model('./model/bert.h5', custom_objects={"AdamWeightDecay": AdamWeightDecay})

def predict(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1)

replacements = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
aspects = {'AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE&OPTIONS',
              'FOOD#PRICES', 'FOOD#QUALITY', 'FOOD#STYLE&OPTIONS', 'LOCATION#GENERAL',
              'RESTAURANT#GENERAL', 'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL'}

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

app = Flask(__name__)
global_predictions = []
data_file = './data/data.json'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    # Đọc dữ liệu từ tệp JSON
    with open('./data/data.json', 'r') as file:
        data = json.load(file)
    return render_template('data.html', predictions = data)

@app.route('/stats')
def stats():
    # Đọc dữ liệu từ tệp JSON
    with open('./data/data.json', 'r') as file:
        data = json.load(file)
    return render_template('stats.html', predictions = data)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    input_text = request.form['text']
    processed_text = text_preprocess(input_text)
    tokenized_input = tokenizer(processed_text, padding='max_length', truncation=True, return_tensors="tf")
    input_ids = tokenized_input['input_ids'].numpy()
    token_type_ids = tokenized_input['token_type_ids'].numpy()
    attention_mask = tokenized_input['attention_mask'].numpy()

    predictions = predict(model, {'input_ids': input_ids,
                                  'token_type_ids': token_type_ids,
                                  'attention_mask': attention_mask})

    results = []
    for aspect, sentiment in zip(aspects, predictions[0]):
        if sentiment:
            results.append({'aspect': aspect, 'sentiment': replacements[sentiment]})

    global_predictions.append({'input_text': input_text, 'predictions': results})

    return jsonify(results)

@app.route('/save_data', methods=['POST'])
def save_data():
    global global_predictions
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            old_data = json.load(file)
    else:
        old_data = []
    old_data.extend(global_predictions)

    with open(data_file, 'w') as file:
        json.dump(old_data, file)

    global_predictions.clear()

    return jsonify(success=True)
if __name__ == '__main__':
    app.run(debug=True)
