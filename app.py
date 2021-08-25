import pymongo
from flask import Flask, render_template, request, url_for, redirect
import pickle
import sklearn
import numpy as np
from itertools import chain




app = Flask(__name__, template_folder='templates')

# These encoders for model3 begins here
#Loding the encoders
# Loading the sex encoder
s_file = open('model3/sex_encoder.pkl', 'rb')
le_sex = pickle.load(s_file)

# Loading the Mother job encoder
mj_file = open('model3/mj_encoder.pkl', 'rb')
le_mj = pickle.load(mj_file)

# Loading the Father job encoder
fj_file = open('model3/fj_encoder.pkl', 'rb')
le_fj = pickle.load(fj_file)

# Loading the Hosteller
h_file = open('model3/h_encoder.pkl', 'rb')
le_h = pickle.load(h_file)

# Loading the Edu_support
es_file = open('model3/es_encoder.pkl', 'rb')
le_es = pickle.load(es_file)

# Loading the Extra paid
ep_file = open('model3/ep_encoder.pkl', 'rb')
le_ep = pickle.load(ep_file)

# Loading the Extracurricular
exc_file = open('model3/exc_encoder.pkl', 'rb')
le_exc = pickle.load(exc_file)

# Loading the higher Education
h_file = open('model3/he_encoder.pkl', 'rb')
le_he = pickle.load(h_file)

# Loading the internet
i_file = open('model3/i_encoder.pkl', 'rb')
le_i = pickle.load(i_file)


# Loading the regression model
model = pickle.load(open('model3/model_ml', 'rb'))

client = pymongo.MongoClient()
mydb = client['mydb']
mycoll = mydb['students_three']


# These encoders for model3 ends here





# These encoders for model4 begins here

# Loading the sex encoder
s_file4 = open('model4/sex_encoder.pkl', 'rb')
le_sex4 = pickle.load(s_file4)

# Loading the Mother job encoder
mj_file4 = open('model4/mj_encoder.pkl', 'rb')
le_mj4 = pickle.load(mj_file4)

# Loading the Father job encoder
fj_file4 = open('model4/fj_encoder.pkl', 'rb')
le_fj4 = pickle.load(fj_file4)

# Loading the Hosteller
h_file4 = open('model4/h_encoder.pkl', 'rb')
le_h4 = pickle.load(h_file4)

# Loading the Edu_support
es_file4 = open('model4/es_encoder.pkl', 'rb')
le_es4 = pickle.load(es_file4)

# Loading the Extra paid
ep_file4 = open('model4/ep_encoder.pkl', 'rb')
le_ep4 = pickle.load(ep_file4)

# Loading the Extracurricular
exc_file4 = open('model4/exc_encoder.pkl', 'rb')
le_exc4 = pickle.load(exc_file4)

# Loading the higher Education
h_file4 = open('model4/he_encoder.pkl', 'rb')
le_he4 = pickle.load(h_file4)

# Loading the internet
i_file4 = open('model4/i_encoder.pkl', 'rb')
le_i4 = pickle.load(i_file4)


# Loading the regression model
model4 = pickle.load(open('model4/model_ml', 'rb'))

client = pymongo.MongoClient()
mydb = client['mydb']
mycol4 = mydb['students_four']

# # These encoders for model4 ends here


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/options")
def options():
    return render_template('options.html')

@app.route("/student_features_entry", methods = ["GET", "POST"])
def student_features_entry():
    if request.method == "POST":
        all_features = list([x for x in request.form.values()])
        #sex = request.form.get('sex')
        #sex_value = int(le_sex.transform([sex]))
        # Fetching all values from form and converting into labled encoder
        student_name = request.form.get('Student_Name_id')
        sex_value = int(le_sex.transform([request.form.get('sex')]))
        age_value = int(request.form.get('age'))
        mj_value = int(le_mj.transform([request.form.get('motherjob')]))
        fj_value = int(le_fj.transform([request.form.get('fatherjob')]))
        travel_value = int(request.form.get('travel'))
        hosteller_value = int(le_h.transform([request.form.get('hosteller')]))
        edu_support_value = int(le_es.transform([request.form.get('educationsupport')]))
        extrapaid_value = int(le_ep.transform([request.form.get('extrapaid')]))
        extracurricular_value = int(le_exc.transform([request.form.get('extracurricular')]))
        highereducation_value = int(le_he.transform([request.form.get('highereducation')]))
        internet_value = int(le_i.transform([request.form.get('internet')]))
        studytime_value = int(request.form.get('studytime'))
        subjectfailed_value = int(request.form.get('subjectfailed'))
        freetime_value = int(request.form.get('freetime'))
        goingout_value = int(request.form.get('goingout'))
        health_value = int(request.form.get('health'))
        tenth_score = request.form.get('tenth_score')
        tenth_score_int = float(tenth_score)
        pu_score = request.form.get('pu_score')
        pu_score_int = float(pu_score)
        first_sem = request.form.get('first_semester')
        first_sem_int = float(first_sem)

        # converting into percentage
        p_first_sem = float((first_sem_int * 10) - 7.5)

        second_sem = request.form.get('second_semester')
        second_sem_int = float(second_sem)

        # converting into percentage
        p_second_sem = float((second_sem_int * 10) - 7.5)

        predict_value = [sex_value, age_value, mj_value, fj_value, travel_value, hosteller_value, studytime_value,subjectfailed_value,
                         edu_support_value, extrapaid_value, extracurricular_value, highereducation_value,
                         internet_value,freetime_value, goingout_value,
                         health_value, tenth_score_int, pu_score_int, p_first_sem, p_second_sem]
        # Getting all the data for prediction and converting into np 2d array
        final_features = [np.array(predict_value)]
        prediction = model.predict(final_features)
        final_prediction = round(prediction[0],2)
        predicted_cgpa = ((float(final_prediction) + 7.5) / 10)
        predicted_cgpa = float("{:.2f}".format(predicted_cgpa))

        if studytime_value==0:
            return render_template('error.html')
        else:
            mycoll.insert_one({"Student_Name": all_features[0], "sex": all_features[1], "Age": all_features[2],
                          "Mothers_Occupation": all_features[3], "Father_Occupation": all_features[4],
                          "Travel_Time": all_features[5], "Hosteller": all_features[6],
                          "Study_Time": all_features[7], "Subject_Failed": all_features[8],
                          "Education_Support": all_features[9], "Extra_Paid": all_features[10],
                          "Extracurricular": all_features[11], "Higher_Education": all_features[12],
                          "Internet": all_features[13], "Free_Time": all_features[14],
                          "Going_Out": all_features[15], "Health": all_features[16],
                          "Tenth_Score": all_features[17], "PU_Score": all_features[18],
                          "First_Semester": all_features[19], "Second_Semester": all_features[20],"Predicted": final_prediction,"Predicted_cgpa": predicted_cgpa})
        return redirect(url_for('output'))

    return render_template('main.html')


@app.route("/student_features_entry_four", methods = ["GET", "POST"])
def student_features_entry_four():
    if request.method == "POST":
        all_features = list([x for x in request.form.values()])
        #sex = request.form.get('sex')
        #sex_value = int(le_sex.transform([sex]))
        # Fetching all values from form and converting into labled encoder
        student_name = request.form.get('Student_Name_id')
        sex_value = int(le_sex4.transform([request.form.get('sex')]))
        age_value = int(request.form.get('age'))
        mj_value = int(le_mj4.transform([request.form.get('motherjob')]))
        fj_value = int(le_fj4.transform([request.form.get('fatherjob')]))
        travel_value = int(request.form.get('travel'))
        hosteller_value = int(le_h4.transform([request.form.get('hosteller')]))
        edu_support_value = int(le_es4.transform([request.form.get('educationsupport')]))
        extrapaid_value = int(le_ep4.transform([request.form.get('extrapaid')]))
        extracurricular_value = int(le_exc4.transform([request.form.get('extracurricular')]))
        highereducation_value = int(le_he4.transform([request.form.get('highereducation')]))
        internet_value = int(le_i4.transform([request.form.get('internet')]))
        studytime_value = int(request.form.get('studytime'))
        subjectfailed_value = int(request.form.get('subjectfailed'))
        freetime_value = int(request.form.get('freetime'))
        goingout_value = int(request.form.get('goingout'))
        health_value = int(request.form.get('health'))
        tenth_score = request.form.get('tenth_score')
        tenth_score_int = float(tenth_score)
        pu_score = request.form.get('pu_score')
        pu_score_int = float(pu_score)
        first_sem = request.form.get('first_semester')
        first_sem_int = float(first_sem)

        # converting into percentage
        p_first_sem = float(((first_sem_int * 10) - 7.5))

        second_sem = request.form.get('second_semester')
        second_sem_int = float(second_sem)

        # converting into percentage
        p_second_sem = float(((second_sem_int * 10) - 7.5))

        third_sem = request.form.get('third_semester')
        third_sem_int = float(third_sem)

        # converting into percentage
        p_third_sem = float(((third_sem_int * 10) - 7.5))


        predict_value = [sex_value, age_value, mj_value, fj_value, travel_value, hosteller_value, studytime_value,subjectfailed_value,
                         edu_support_value, extrapaid_value, extracurricular_value, highereducation_value,
                         internet_value,freetime_value, goingout_value,
                         health_value, tenth_score_int, pu_score_int, p_first_sem, p_second_sem, p_third_sem]
        # Getting all the data for prediction and converting into np 2d array
        final_features = [np.array(predict_value)]
        prediction = model4.predict(final_features)
        final_prediction = round(prediction[0],2)
        predicted_cgpa = ((float(final_prediction) + 7.5) / 10)
        predicted_cgpa = float("{:.2f}".format(predicted_cgpa))
        if studytime_value==0:
            return render_template('error1.html')
        else:
            mycol4.insert_one({"Student_Name": all_features[0], "sex": all_features[1], "Age": all_features[2],
                          "Mothers_Occupation": all_features[3], "Father_Occupation": all_features[4],
                          "Travel_Time": all_features[5], "Hosteller": all_features[6],
                          "Study_Time": all_features[7], "Subject_Failed": all_features[8],
                          "Education_Support": all_features[9], "Extra_Paid": all_features[10],
                          "Extracurricular": all_features[11], "Higher_Education": all_features[12],
                          "Internet": all_features[13], "Free_Time": all_features[14],
                          "Going_Out": all_features[15], "Health": all_features[16],
                          "Tenth_Score": all_features[17], "PU_Score": all_features[18],
                          "First_Semester": all_features[19], "Second_Semester": all_features[20],"Third_Semester": all_features[21],"Predicted": final_prediction, "Predicted_cgpa": predicted_cgpa})
        return redirect(url_for('output4'))

    return render_template('main4.html')



@app.route("/output4", methods = ["GET", "POST"])
def output4():
    res_list = []
    if request.method == "POST":
        student_name = request.form.get('Student_Name_id')
        result_con = []
        result = mycol4.find({'Student_Name': student_name})
        for i in result:
            result_con.append(i)
        for idx, sub in enumerate(result_con, start=0):
            if idx == 1:
                res_list.append(list(sub.values()))
            else:
                res_list.append(list(sub.values()))
    flatten_list = list(chain.from_iterable(res_list))
    #print(flatten_list[23])
    #predicted_cgpa = ((float(flatten_list[23]) + 7.5) / 10)
    #predicted_cgpa = float("{:.2f}".format(predicted_cgpa))
    student_names_list = mycol4.find({}, {"Student_Name": 1})
    return render_template('output4.html', student_names_list=student_names_list, flatten_list=flatten_list)



@app.route("/output", methods = ["GET", "POST"])
def output():
    res_list = []
    #pre_data = {}
    if request.method == "POST":
        student_name = request.form.get('Student_Name_id')
        result_con = []
        result = mycoll.find({'Student_Name': student_name})
        #pre_data = mycoll.find({'Student_Name': student_name}, {'Predicted': 1})
        for i in result:
            result_con.append(i)
        #dictionary convert into list(fetching 1st value based on student name)
        #[] remove
        for idx, sub in enumerate(result_con, start=0):
            if idx == 1:
                res_list.append(list(sub.values()))
            else:
                res_list.append(list(sub.values()))

    # Converting into proper list
    flatten_list = list(chain.from_iterable(res_list))
    #pre_new = dict(pre_data)
    #print(pre_new)
    #predicted_cgpa = ((float(flatten_list[22]) + 7.5) / 10)
    #predicted_cgpa = float("{:.2f}".format(predicted_cgpa))
    student_names_list = mycoll.find({},{"Student_Name": 1})
    return render_template('output.html', student_names_list = student_names_list,flatten_list = flatten_list)



if __name__ == "__main__":
    app.run(host='192.168.147.1',debug=True)