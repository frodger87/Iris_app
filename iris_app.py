import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn import datasets

#формирование боковой панели
st.sidebar.header('Укажите ваши параметры')


def user_input_features():
    """Берет данные, введенные пользователем с помощью ползунков,
    и превращает в датасет
    """
    # st.sidebar.slider(название характерикстики, min начение ползунка, max значение ползунка, значение ползунка
    # по умолчанию при загрузке страницы)
    sepal_lenght = st.sidebar.slider('Длинна чашелистика', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Ширина чашелистика', 2.0, 4.4, 3.4)
    petal_lenght = st.sidebar.slider('Длинна лепестка', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Ширина лепестка', 0.1, 2.5, 0.2)

    data = {'sepal_lenght': sepal_lenght,
            'sepal_width': sepal_width,
            'petal_lenght': petal_lenght,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

#загружаем датасет ирис
iris = datasets.load_iris()
#выделяем признаки и таргет
X = iris.data
y = iris.target
#объявляем модель
svc = SVC(C= 3.6616809069626406, random_state=1, probability=True)
#обучаем модель
svc.fit(X, y)
#получение предсказание с помощью обученной модели
prediction = svc.predict(df)
#получение сведений о прогностической вероятности
prediction_proba = svc.predict_proba(df)
#заголовок в приложении
st.write("""
#  Простое приложение для классификации цветков Ириса.""")
#первый подзаголовок в основной панели
st.subheader('Параметры вводимые пользователем')
#выводим на основную панель содержимое df
st.write(df)
#второй подзаголовок основной панели
st.subheader('Метка класса и соотвествущий номер')
#выводятся названия видов цветков и соответствующие им номера
st.write(iris.target_names)
#третий подзагловок основной панели
st.subheader('Прогноз')
#вывод результата классификации
st.write(iris.target_names[prediction])
#четвертый подзаголовок основной панели
st.subheader('Вероятности прогноза')
#вывод данных о прогностической вероятности
st.write(prediction_proba)
