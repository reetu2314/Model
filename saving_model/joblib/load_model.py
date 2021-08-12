import joblib

# load the model

model = joblib.load('diabetic_79.pkl')

# names = ['preg' , 'plas' , 'pres' , 'skin' , 'test' , 'mass' , 'pedi' , 'age' , 'class']
result = model.predict([[1,1,1,1,1,1,1,1]])
print(result)

if result[0] == 1:
    print('person is diabetic')
else:
    print('person is not diabetic')
