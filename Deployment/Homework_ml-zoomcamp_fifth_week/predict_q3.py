import pickle


model_file = 'model1.bin'
data_encoder = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(data_encoder, 'rb') as f_in:
    dv = pickle.load(f_in)

customer_id = "0001"

customer1 = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': round(float(y_pred),3),
        'churn': bool(churn)
    }

    return result

prediction = predict(customer1)

print(prediction)

if prediction['churn'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)

