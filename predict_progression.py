import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve,f1_score, accuracy_score,classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from scipy.stats import mannwhitneyu,chi2_contingency
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load

import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def process_data(input_data, model_path="best_modelokt_clinic.h5", pipeline_path="best_pipeline_okt_clinic.joblib"):
    pts = read_file(input_data)
    prediction = make_prediction(pts, model_path, pipeline_path)
    return prediction
    
    
def read_file(input_data):
    pts = pd.read_excel(input_data)
    pts.drop('Категория', axis=1, inplace=True)
    pts['Значение'] = pts['Значение'].apply(lambda x: 1 if x == 'да' else (0 if x == 'нет' else x))
    pts = pts.set_index('Характеристика').T
    dic = {
    'Продолжительность РС (года)': 'Продолжительность заболевания (года)',
    'Высокоактивный РС (да/нет)': 'ВАРС (1 - да, 0- нет)',
    'Новые очаги или накопление контраста на МРТ (да/нет)': 'появления новых очагов на МРТ-изображениях (нет новых или увеличенных очагов на Т2-взвешенных изображениях или очагов, накапливающих контраст на Т1-взвешенных изображениях)',
    'Баллы за тест MoCA': 'MоCA\t',
    'Баллы за тест SDMT': 'визуальный тест (дата 1 осмотра)',
    'Баллы за тест EQ-5D': 'ЕQ-5D        ',
    'Время выполнения 25-футового теста ходьбы (сек)': 'average leg',
    'Время выполнения 9-колышкового теста правой рукой (сек)': 'average right',
    'Время выполнения 9-колышкового теста левой рукой (сек)': 'average left',
    'Баллы за зрительные нарушения по шкале Курцке (EDSS)': 'зрение',
    'Баллы за стволовые нарушения по шкале Курцке (EDSS)': 'Ствол',
    'Баллы за пирамидные нарушения по шкале Курцке (EDSS)': 'Пирамид',
    'Баллы за мозжечковые нарушения по шкале Курцке (EDSS)': 'Мозж',
    'Баллы за чувствительные нарушения по шкале Курцке (EDSS)': 'Чувств',
    'Баллы за тазовые нарушения по шкале Курцке (EDSS)': 'Наруш таз орг',
    'Баллы за когнитивные нарушения по шкале Курцке (EDSS)': 'интелект',
    'Когнитивные нарушения (нет/легкие/умеренные/выраженные)': 'когнитивные нарушения',
    'У пациента есть сердечно-сосудистые заболевания (да/нет)': 'Сердечно-сосудистые заболевания\nда– 1 \nнет-0',
    'У пациента есть другие аутоиммунные заболевания(да/нет)': 'Аутоиммунные заболевания\nда– 1 \nнет-0',
    'У пациента зафиксированы аллергические реакции (да/нет)': 'Отягощенный аллергологический анамнез\nДа – 1 \nНет – 0 \n',
    'Толщина перипапиллярного слоя нервных волокон сетчатки (pRNFL), OS': 'pRNFL total OS',
    'Толщина перипапиллярного слоя нервных волокон сетчатки в нижнем квадранте (pRNFL), OS': 'pRNFL inf OS',
    'Толщина слоя ганглиозных клеток (GCL+), OS': 'Total GCL+, um, OS',
    'Толщина слоя ганглиозных клеток в нижнем квадранте (GCL+), OD': 'Inferior GCL+, um, OD',
    'Толщина слоя ганглиозных клеток в нижнем квадранте (GCL+), OS': 'Inferior GCL+, um, OS',
    'Толщина слоя ганглиозных клеток в верхнем квадранте (GCL+), OS': 'Superior GCL+, um, OS',
    'Толщина комплекса слоя ганглиозных клеток и внутреннего плексиформного слоя (GCL++), OS': 'Total GCL++, um, OS',
    'Толщина комплекса слоя ганглиозных клеток и внутреннего плексиформного слоя в нижнем квадранте (GCL++), OS': 'Inferior GCL++,um, OS'
    }

    pts.rename(columns=dic, inplace=True)
    pts['когнитивные нарушения'] = pts['когнитивные нарушения'].apply(lambda x: 3 if x == 'выраженные' else 
                                                                      (1 if x == 'легкие' else (2 if x == 'умеренные' else 0)))
    return pts

def make_prediction(pts, model_path, pipeline_path):
    cols = [
        'Продолжительность заболевания (года)',
        'ВАРС (1 - да, 0- нет)',
        'появления новых очагов на МРТ-изображениях (нет новых или увеличенных очагов на Т2-взвешенных изображениях или очагов, накапливающих контраст на Т1-взвешенных изображениях)',
        'MоCA\t',
        'зрение',
        'ЕQ-5D        ',
        'Total GCL++, um, OS',
        'Inferior GCL++,um, OS',
        'Чувств',
        'pRNFL total OS',
        'Inferior GCL+, um, OD',
        'Inferior GCL+, um, OS',
        'Отягощенный аллергологический анамнез\nДа – 1 \nНет – 0 \n',
        'Наруш таз орг',
        'Ствол',
        'визуальный тест (дата 1 осмотра)',
        'интелект',
        'Сердечно-сосудистые заболевания\nда– 1 \nнет-0',
        'pRNFL inf OS',
        'average left',
        'когнитивные нарушения',
        'Мозж',
        'Total GCL+, um, OS',
        'average right',
        'Пирамид',
        'average leg',
        'Superior GCL+, um, OS',
        'Аутоиммунные заболевания\nда– 1 \nнет-0'
    ]
    pts = pts[cols]
    pipeline = load(pipeline_path)
    model = load_model(model_path)
    pts_nn = pipeline.transform(pts)
    y_pred_proba = model.predict(pts_nn).flatten()
    prediction = (y_pred_proba >= 0.5).astype(int)
    if prediction == 1:
        return 'Есть вероятность прогрессирования'
    else:
        return 'Вероятность прогрессирования не обнаружена'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict disease progression")
    parser.add_argument('-model', type=str, required=True, help="Path to the model file")
    parser.add_argument('-pipeline', type=str, required=True, help="Path to the pipeline file")
    parser.add_argument('-p', type=str, required=True, help="Path to the patient Excel file")
    
    args = parser.parse_args()
    
    prediction = process_data(args.p, args.model, args.pipeline)
    print(prediction)



