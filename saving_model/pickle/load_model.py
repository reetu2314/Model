import pickle

# load the model

model = pickle.load(open('diabetic_79.sav' , 'rb'))

# names = ['preg' , 'plas' , 'pres' , 'skin' , 'test' , 'mass' , 'pedi' , 'age' , 'class']
result = model.predict([[1,1,1,1,1,1,1,1]])
print(result)

# if result[0] == 1:
#     print('person is diabetic')
# else:
#     print('person is not diabetic')