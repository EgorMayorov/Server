import os
import numpy as np
import pandas as pd
import io
from flask import Flask, render_template, url_for, send_from_directory, redirect, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
import time
from wtforms import StringField, FloatField, IntegerField, SelectField
from wtforms.validators import Optional
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from ensembles import RandomForestMSE, GradientBoostingMSE

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'
app.url_map.strict_slashes = False

# paramsdf = pd.DataFrame()
# data_train = pd.DataFrame()
data_test = pd.DataFrame()
data_val = pd.DataFrame()
# target_name = ''


def forget_data():
    global data_train
    global data_test
    global target_name
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    data_val = pd.DataFrame()
    target_name = ''


@app.route('/', methods=['GET', 'POST'])
def start():
    forget_data()
    return render_template('start.html', bg_class="bg")


@app.route('/model_choose', methods=['GET', 'POST'])
def model_choose():
    return render_template('model_choose.html')


class ParamsForm(FlaskForm):
    n_estimators = IntegerField('Количество деревьев')
    learning_rate = FloatField('Темп обучения')
    max_depth = IntegerField('Максимальная глубина')
    feature_num = IntegerField('Число признаков для дерева')


@app.route('/params_GB', methods=['GET', 'POST'])
def GB_params():
    params = ParamsForm(meta={'csrf': False})
    if params.validate_on_submit():
        global paramsdf
        paramsdf = pd.DataFrame({'Параметр': ['Количество деревьев',
                                              'Темп обучения',
                                              'Максимальная глубина',
                                              'Число признаков для дерева'],
                                 'Значение': [params.n_estimators.data,
                                              params.learning_rate.data,
                                              params.max_depth.data,
                                              params.feature_num.data]})
        global model
        model = GradientBoostingMSE(params.n_estimators.data, params.learning_rate.data,
                                    params.max_depth.data, params.feature_num.data)
        return redirect(url_for('data_input'))
    return render_template('GB_params.html', params=params)


@app.route('/params_RF', methods=['GET', 'POST'])
def RF_params():
    params = ParamsForm(meta={'csrf': False})
    if params.validate_on_submit():
        del params.learning_rate
        global paramsdf
        paramsdf = pd.DataFrame({'Параметр': ['Количество деревьев',
                                              'Максимальная глубина',
                                              'Число признаков для дерева'],
                                 'Значение': [params.n_estimators.data,
                                              params.max_depth.data,
                                              params.feature_num.data]})
        global model
        model = RandomForestMSE(params.n_estimators.data, params.max_depth.data, params.feature_num.data)
        return redirect(url_for('data_input'))
    return render_template('RF_params.html', params=params)


class DataForm(FlaskForm):
    train_data = FileField('Обучающий датасет', validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    target_name = StringField('Целевая переменная')
    val_data = FileField('Валидационная выборка', validators=[Optional(), FileAllowed(['csv'], 'Неверный формат')])
    test_data = FileField('Датасет для предсказания', validators=[Optional(), FileAllowed(['csv'], 'Неверный формат')])


@app.route('/data', methods=['GET', 'POST'])
def data_input():
    data = DataForm()
    if data.validate_on_submit():
        global data_train
        global data_test
        global data_val
        global target_name
        data_train = pd.read_csv(data.train_data.data, index_col=0)
        if data.val_data.data is not None:
            data_val = pd.read_csv(data.val_data.data, index_col=0)
        if data.test_data.data is not None:
            data_test = pd.read_csv(data.test_data.data, index_col=0)
        target_name = data.target_name.data
        return redirect(url_for('presentation'))
    return render_template('data_input.html', data=data)


@app.route('/presentation', methods=['GET', 'POST'])
def presentation():
    return render_template('presentation.html', params=paramsdf, data_train=data_train.head(),
                           ch1=data_test.empty, data_test=data_test.head(), target_name=target_name,
                           ch2=data_val.empty, data_val=data_val.head())


@app.route('/trained_model', methods=['GET', 'POST'])
def trained_model():
    global model
    global data_train
    global data_test
    global data_val
    global target_name
    X_train = data_train.drop(columns=target_name).to_numpy()
    Y_train = data_train[target_name].to_numpy()
    global history
    if not data_val.empty:
        X_val = data_val.drop(columns=target_name).to_numpy()
        Y_val = data_val[target_name].to_numpy()
        history = model.fit(X_train, Y_train, X_val, Y_val, hist=True)
    else:
        history = model.fit(X_train, Y_train, hist=True)
    best_loss = np.amin(history['rmse_train'])
    time = np.sum(history['time'])
    return render_template('trained_model.html', time=round(time, 3), best_loss=round(best_loss, 3), ch=data_test.empty)


@app.route('/plot')
def plot():
    global history
    global paramsdf
    n_estimators = paramsdf['Значение'].values[0]
    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    plt.suptitle('Поведение алгоритма случайный лес в зависимости от количества деревьев', fontsize=15)

    plt.subplot(2, 1, 1)
    plt.title('Значения RMSE')
    plt.xlabel('Количество деревьев')
    plt.ylabel('RMSE')
    plt.plot(np.arange(1, n_estimators + 1), history['rmse_train'], color='green', linestyle='-',
             linewidth=3, label='Обучающая выборка')
    if history['rmse_val']:
        plt.plot(np.arange(1, n_estimators + 1), history['rmse_val'], color='gold', linestyle='-',
                 linewidth=3, label='Валидационная выборка')
    plt.legend()
    plt.grid(which='major', color='grey', linewidth=0.5)

    plt.subplot(2, 1, 2)
    plt.title('Зависимость времени работы от количества деревьев')
    plt.xlabel('Количество деревьев')
    plt.ylabel('Время работы, с')
    plt.plot(np.arange(1, n_estimators + 1), np.cumsum(history['time']), color='deeppink', linestyle='-',
             linewidth=3, label='Время')
    plt.grid(which='major', color='grey', linewidth=0.5)

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


class Save(FlaskForm):
    save = SelectField(' ', validators=[Optional()], choices=[('да', 'да')])


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global model
    global data_test
    data = data_test.to_numpy()
    global pred
    start = time.time()
    pred = model.predict(data)
    end = time.time() - start
    pred = np.around(pred, decimals=2)
    global target_name
    pred = pd.DataFrame({target_name: pred})
    save = Save(meta={'csrf': False})
    if save.validate_on_submit():
        filename = 'prediction.csv'
        path = os.path.join(os.getcwd(), 'tmp/')
        if not os.path.exists(path):
            os.mkdir(path)
        pred.to_csv(os.path.join(path, filename))
        return send_from_directory(path, filename, as_attachment=True)
    return render_template('prediction.html', data=pred, time=round(end, 3), save=save)
