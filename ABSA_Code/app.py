import re
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from transformers import TFRobertaModel, AutoTokenizer, AdamWeightDecay
from preprocess import text_preprocess
import json

tf.keras.utils.get_custom_objects().update({'TFRobertaModel': TFRobertaModel})
tf.keras.utils.get_custom_objects().update({'AdamWeightDecay': AdamWeightDecay})
model = tf.keras.models.load_model('./model/bert.h5', custom_objects={"AdamWeightDecay": AdamWeightDecay})

def predict(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1)

replacements = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
categories = {'AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE&OPTIONS',
              'FOOD#PRICES', 'FOOD#QUALITY', 'FOOD#STYLE&OPTIONS', 'LOCATION#GENERAL',
              'RESTAURANT#GENERAL', 'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL'}

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

app = Flask(__name__)
global_predictions = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html', predictions=global_predictions)

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
    for category, sentiment in zip(categories, predictions[0]):
        if sentiment:
            results.append({'category': category, 'sentiment': replacements[sentiment]})

    global_predictions.append({'input_text': input_text, 'predictions': results})

    return jsonify(results)

@app.route('/save_data', methods=['POST'])
def save_data():
    # Lưu dữ liệu vào tệp JSON
    with open('./data/data.json', 'w') as file:
        json.dump(global_predictions, file)

    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)