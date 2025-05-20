import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import numpy as np

def analysis_and_model_page():
    st.title("Анализ и Модель")
    
    # Initialize session state
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'selected_model_name' not in st.session_state:
        st.session_state.selected_model_name = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = LabelEncoder()
    if 'scaler' not in st.session_state:
        st.session_state.scaler = StandardScaler()

    # Data loading options
    st.header("Загрузка Датасета")
    st.info("Загружаемый CSV должен содержать столбцы: Тип, Температура воздуха [K], Температура процесса [K], Скорость вращения [об/мин], Крутящий момент [Нм], Износ инструмента [мин], Отказ машины")
    data_source = st.radio("Выберите источник данных:", ("Загрузить CSV", "Получить из репозитория UCI"))
    
    data = None
    if data_source == "Загрузить CSV":
        uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Ошибка чтения CSV: {e}")
    else:
        try:
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            st.write("Исходные столбцы датасета UCI:", data.columns.tolist())
            column_mapping = {
                'Air temperature': 'Air temperature [K]',
                'Process temperature': 'Process temperature [K]',
                'Rotational speed': 'Rotational speed [rpm]',
                'Torque': 'Torque [Nm]',
                'Tool wear': 'Tool wear [min]',
                'Target': 'Machine failure',
                'Air_temperature': 'Air temperature [K]',
                'Process_temperature': 'Process temperature [K]',
                'Rotational_speed': 'Rotational speed [rpm]',
                'Tool_wear': 'Tool wear [min]',
                'Air temperature K': 'Air temperature [K]',
                'Process temperature K': 'Process temperature [K]'
            }
            data = data.rename(columns=column_mapping)
            st.write("Переименованные столбцы датасета UCI:", data.columns.tolist())
        except Exception as e:
            st.error(f"Ошибка получения датасета: {e}")
            return
    
    if data is not None:
        st.write("Датасет успешно загружен")
        st.write(data.head())
        
        # Validate required columns
        required_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"В датасете отсутствуют следующие обязательные столбцы: {missing_columns}")
            st.write("Доступные столбцы:", data.columns.tolist())
            return
        
        # Preprocessing
        st.header("Предобработка Данных")
        columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        
        data = data.rename(columns={
            'Air temperature [K]': 'Air_temperature_K',
            'Process temperature [K]': 'Process_temperature_K',
            'Rotational speed [rpm]': 'Rotational_speed_rpm',
            'Torque [Nm]': 'Torque_Nm',
            'Tool wear [min]': 'Tool_wear_min'
        })
        
        data['Type'] = st.session_state.label_encoder.fit_transform(data['Type'])
        
        st.write("Пропущенные значения:")
        st.write(data.isnull().sum())
        
        numerical_features = ['Air_temperature_K', 'Process_temperature_K', 
                             'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        data[numerical_features] = st.session_state.scaler.fit_transform(data[numerical_features])
        
        X = data.drop('Machine failure', axis=1)
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model training
        st.header("Обучение Модели")
        models = {
            "Логистическая Регрессия": LogisticRegression(random_state=42, C=0.05, class_weight='balanced', max_iter=1000),
            "Случайный Лес": RandomForestClassifier(n_estimators=10, max_depth=1, min_samples_split=100, 
                                                   min_samples_leaf=50, random_state=42, class_weight='balanced'),
            "XGBoost": XGBClassifier(n_estimators=10, learning_rate=0.01, max_depth=1, reg_lambda=10.0, 
                                    reg_alpha=5.0, random_state=42, 
                                    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), 
                                    eval_metric='logloss'),
            "SVM": SVC(kernel='rbf', C=0.05, gamma='scale', random_state=42, probability=True, class_weight='balanced')
        }
        
        selected_model = st.selectbox("Выберите модель", list(models.keys()))
        if st.button("Обучить модель"):
            model = models[selected_model]
            try:
                # Borderline-SMOTE + undersampling
                pipeline = Pipeline([
                    ('smote', BorderlineSMOTE(sampling_strategy=1.0, k_neighbors=3, random_state=42)),
                    ('undersample', RandomUnderSampler(sampling_strategy=1.0, random_state=42))
                ])
                X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
                
                # Calibrate with sigmoid
                model = CalibratedClassifierCV(model, method='sigmoid', cv=15)
                model.fit(X_train_res, y_train_res)
                st.session_state.trained_model = model
                st.session_state.selected_model_name = selected_model
                
                # Model evaluation
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                st.header("Результаты Оценки Модели")
                st.write(f"**Точность**: {accuracy:.2f}")
                st.write(f"**ROC-AUC**: {roc_auc:.2f}")
                st.write("**Отчет по классификации**:")
                st.text(class_report)
                
                # Confusion matrix
                st.subheader("Матрица Ошибок")
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Предсказанный класс')
                ax.set_ylabel('Фактический класс')
                st.pyplot(fig)
                
                # ROC curve
                st.subheader("Кривая ROC")
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'{selected_model} (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
                ax.set_xlabel('Доля ложноположительных')
                ax.set_ylabel('Доля истинноположительных')
                ax.set_title('Кривая ROC')
                ax.legend()
                st.pyplot(fig)
                
                # Calibration curve
                st.subheader("Кривая Калибровки")
                prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
                fig, ax = plt.subplots()
                ax.plot(prob_pred, prob_true, marker='o', label=selected_model)
                ax.plot([0, 1], [0, 1], linestyle='--', label='Идеальная калибровка')
                ax.set_xlabel('Предсказанная вероятность')
                ax.set_ylabel('Истинная вероятность')
                ax.legend()
                st.pyplot(fig)
                
                # Probability histogram
                st.subheader("Распределение Предсказанных Вероятностей")
                fig, ax = plt.subplots()
                sns.histplot(y_pred_proba, bins=20, ax=ax)
                ax.set_xlabel("Предсказанная вероятность отказа")
                ax.set_ylabel("Количество")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка обучения модели: {e}")
        
        # Prediction interface
        st.header("Прогнозирование Отказа Машины")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для прогноза:")
            type_input = st.selectbox("Тип", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха [K]", min_value=0.0, value=300.0)
            process_temp = st.number_input("Температура процесса [K]", min_value=0.0, value=310.0)
            rotational_speed = st.number_input("Скорость вращения [об/мин]", min_value=0, value=1500)
            torque = st.number_input("Крутящий момент [Нм]", min_value=0.0, value=40.0)
            tool_wear = st.number_input("Износ инструмента [мин]", min_value=0, value=0)
            submit_button = st.form_submit_button("Спрогнозировать")
            
            if submit_button:
                if st.session_state.trained_model is None or st.session_state.selected_model_name != selected_model:
                    st.error("Сначала обучите модель перед выполнением прогноза!")
                else:
                    try:
                        # Prepare input data
                        input_data = pd.DataFrame({
                            'Type': [st.session_state.label_encoder.transform([type_input])[0]],
                            'Air_temperature_K': [air_temp],
                            'Process_temperature_K': [process_temp],
                            'Rotational_speed_rpm': [rotational_speed],
                            'Torque_Nm': [torque],
                            'Tool_wear_min': [tool_wear]
                        })
                        input_data[numerical_features] = st.session_state.scaler.transform(input_data[numerical_features])
                        
                        # Make prediction
                        prediction = st.session_state.trained_model.predict(input_data)
                        prediction_proba = st.session_state.trained_model.predict_proba(input_data)
                        
                        # Strong smoothing
                        smoothed_proba = (prediction_proba + 1e-3) / (1 + 2e-3)
                        failure_proba = smoothed_proba[0, 1]
                        
                        # Dynamic warning threshold
                        temp_diff = process_temp - air_temp
                        power = torque * rotational_speed * (2 * 3.14159 / 60)
                        tool_torque = tool_wear * torque
                        is_safe = (tool_wear < 200 and temp_diff > 8.6 and 3500 <= power <= 9000 and tool_torque < 11000)
                        threshold = 0.0001 if is_safe else 0.01
                        
                        # Display results
                        st.write(f"**Прогноз**: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                        st.write(f"**Вероятность отказа**: {failure_proba:.3f}")
                        
                        # Warn about extreme probabilities
                        if failure_proba < threshold or failure_proba > (1 - threshold):
                            st.warning("Модель очень уверена в этом прогнозе. Это может указывать на переобучение или проблемы с данными. Проверьте входные значения и рассмотрите переобучение модели.")
                    except Exception as e:
                        st.error(f"Ошибка выполнения прогноза: {e}")