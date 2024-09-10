from flask import Flask, request, redirect, url_for, session, render_template, make_response, jsonify, flash
import mysql.connector
import bcrypt
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
from io import BytesIO, StringIO
import zipfile
import base64
from werkzeug.utils import secure_filename
from os import path
import matplotlib.pyplot as plt
import io
import logging


app = Flask(__name__)
app.secret_key = 'creditscore'
ALLOWED_EXTENSIONS = {"csv"}
logging.basicConfig(level=logging.DEBUG)

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='score'
    )
    return connection
@app.route("/", methods=["GET"])
def root():
    return render_template("index.html")
    
@app.route("/home", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cccd = request.form['cccd']
        password = request.form['password']
        session['cccd'] = cccd
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE cccd = %s", (cccd,))
        user = cursor.fetchone()
        connection.close()

        if user and bcrypt.checkpw(password.encode('utf-8'),user[3].encode('utf-8')):
            session['user_id'] = user[0]  # ID người dùng
            session['cccd'] = cccd  # CCCD người dùng
            session['role'] = user[6]  # Vai trò người dùng (user/admin)
            logging.debug(f"Logged in as: {session['role']}")  # Sử dụng logging
            # Chuyển hướng đến trang yêu cầu vay
            if session['role'] == 'nhanvien':
                flash("Logged in successfully!")
                return redirect(url_for('nhanvien'))
            elif session['role'] == 'admin':
                flash("Logged in successfully!")
                return redirect(url_for('admin_employees'))
            else:
                flash("Logged in successfully!")
                return redirect(url_for('loan_request'))

        else:
            flash("Invalid cccd or password.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        cccd = request.form['cccd']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash("Passwords do not match. Please try again.")
            return redirect(url_for('register'))
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        name = request.form['name']
        phone = request.form['phone']
        address = request.form['address']
        
        # hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        connection = get_db_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (cccd, password, name, phone, address, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (cccd, hashed_password.decode('utf-8'), name, phone, address, 'user'))
            connection.commit()
            connection.close()
            flash("Registration successful. Please log in.")
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            connection.close()
            return f"Error: {err}"

    return render_template('register.html')

@app.route('/admin/employees')
def admin_employees():
    if 'cccd' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE role = 'nhanvien'")
    employees = cursor.fetchall()
    cursor.close()
    connection.close()

    return render_template('admin_employees.html', employees=employees)
@app.route('/admin/employee/add', methods=['GET', 'POST'])
def admin_add_employee():
    if 'cccd' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form['name']
        cccd = request.form['cccd']
        phone = request.form['phone']
        address = request.form['address']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users (name, cccd, phone, address, role, password) VALUES (%s, %s, %s, %s, %s, %s)",
            (name, cccd, phone, address, 'nhanvien', hashed_password)
        )
        connection.commit()
        cursor.close()
        connection.close()

        return redirect(url_for('admin_employees'))

    return render_template('admin_add_employee.html')

@app.route('/admin/employee/edit/<int:id>', methods=['GET', 'POST'])
def admin_edit_employee(id):
    if 'cccd' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    if request.method == 'POST':
        name = request.form['name']
        cccd = request.form['cccd']
        phone = request.form['phone']
        address = request.form['address']
        cursor.execute(
            "UPDATE users SET name = %s, cccd = %s, phone = %s, address = %s WHERE id = %s",
            (name, cccd, phone, address, id)
        )
        connection.commit()
        cursor.close()
        connection.close()

        return redirect(url_for('admin_employees'))

    cursor.execute("SELECT * FROM users WHERE id = %s", (id,))
    employee = cursor.fetchone()
    cursor.close()
    connection.close()

    return render_template('admin_edit_employee.html', employee=employee)

@app.route('/admin/employee/delete/<int:id>', methods=['POST'])
def admin_delete_employee(id):
    if 'cccd' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (id,))
    connection.commit()
    cursor.close()
    connection.close()

    return redirect(url_for('admin_employees'))

@app.route("/credit", methods=["POST", "GET"])
def credit():
    if request.method == "POST":
        LOAN = float(request.form.get("LOAN"))
        MORTDUE = float(request.form.get("MORTDUE"))
        VALUE = float(request.form.get("VALUE"))
        REASON = request.form.get("REASON")
        JOB = request.form.get("JOB")
        YOJ = float(request.form["YOJ"])
        DEROG = float(request.form.get("DEROG"))
        DELINQ = float(request.form.get("DELINQ"))
        CLAGE = float(request.form.get("CLAGE"))
        NINQ = float(request.form.get("NINQ"))
        CLNO = float(request.form.get("CLNO"))
        DEBTINC = float(request.form.get("DEBTINC"))

        input_data = {
            "LOAN": LOAN,
            "MORTDUE": MORTDUE,
            "VALUE": VALUE,
            "REASON": REASON,
            "JOB": JOB,
            "YOJ": YOJ,
            "DEROG": DEROG,
            "DELINQ": DELINQ,
            "CLAGE": CLAGE,
            "NINQ": NINQ,
            "CLNO": CLNO,
            "DEBTINC": DEBTINC
        }

        # Chuyển đổi dữ liệu thành DataFrame và tiền xử lý
        input_df = pd.DataFrame([input_data])
        input_df_temp = preProcessData(input_df)
        pred_score = model.predict_proba(input_df_temp)[:, 1]
        score = credit_score(pred_score[0])
        risk = credit_score_category(score)

        return redirect(url_for("show_result", score=score, risk=risk))

    return render_template("credit_form.html")
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/analyse_file", methods=["POST", "GET"])
def analyse_file():
    uploaded_files = request.files.getlist("files[]")
    uploaded_files_df = dict()

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            uploaded_filename = secure_filename(file.filename)
            uploaded_csv_file = pd.read_csv(file)

            processed_input = preProcessData(uploaded_csv_file)
            temp_result = model.predict(processed_input)
            pred_scores = model.predict_proba(processed_input)[
                :, 1
            ]  # Xác suất của lớp 'BAD'

            scores = [credit_score(p) for p in pred_scores]

            uploaded_csv_file["Prediction"] = [
                "GOOD" if i == 0 else "BAD" for i in temp_result
            ]
            uploaded_csv_file["Score"] = scores
            uploaded_csv_file["Risk"] = pd.cut(
                scores,
                bins=[-float("inf"), 580, 670, 740, 800, float("inf")],
                labels=["High", "Medium-High", "Medium","Medium-Low","Low"],
            )

            uploaded_files_df[uploaded_filename] = uploaded_csv_file
        else:
            return jsonify(success=False, error="Only CSV Allowed!!")

    # Lấy kết quả từ tệp đầu tiên để hiển thị
    for filename, dataframe in uploaded_files_df.items():
        data = dataframe.to_dict(orient="records")
        column_names = dataframe.columns.tolist()
        break

    return render_template("result2.html", data=data, column_names=column_names)


@app.route("/download_predict_data", methods=["POST", "GET"])
def download_predict_data():
    uploaded_files = request.files.getlist("files[]")
    uploaded_files_df = dict()

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            uploaded_filename = secure_filename(file.filename)
            uploaded_csv_file = pd.read_csv(file)

            processed_input = preProcessData(uploaded_csv_file)
            temp_result = model.predict(processed_input)
            pred_scores = model.predict_proba(processed_input)[
                :, 1
            ]  # Xác suất của lớp 'BAD'

            scores = [credit_score(p) for p in pred_scores]

            uploaded_csv_file["Prediction"] = [
                "GOOD" if i == 0 else "BAD" for i in temp_result
            ]
            uploaded_csv_file["Score"] = scores

            uploaded_files_df[uploaded_filename] = uploaded_csv_file
        else:
            return jsonify(success=False, error="Only CSV Allowed!!")

    return download_files(uploaded_files_df)
def download_files(uploaded_files_df):
    # if multiple files are there, zip and return files
    if len(uploaded_files_df) > 1:
        b_buffer = BytesIO()
        with zipfile.ZipFile(b_buffer, "w") as zip:
            for fName, dataframe in uploaded_files_df.items():
                zip.writestr(
                    fName.rsplit(".", 1)[0] + "-new.csv", dataframe.to_csv(index=False)
                )
        zip.close()
        b_buffer.seek(0)

        response = make_response(b_buffer.getvalue())
        cd_headers = "attachment; filename=csv-multiple-" + ".zip"
        response.headers["Content-Disposition"] = cd_headers
        response.mimetype = "zip"
        b_buffer.close()

        return response
    else:
        filename = ""
        s_buffer = StringIO()
        for fName, dataframe in uploaded_files_df.items():
            dataframe.to_csv(s_buffer, index=False)
            filename = fName.rsplit(".", 1)[0] + "-" + "-new.csv"

        b_buffer = BytesIO()
        b_buffer.write(s_buffer.getvalue().encode("utf-8"))
        b_buffer.seek(0)
        s_buffer.close()

        response = make_response(b_buffer.getvalue())
        cd_headers = "attachment; filename={}".format(filename)
        response.headers["Content-Disposition"] = cd_headers
        response.mimetype = "text/csv"
        b_buffer.close()

        return response
@app.route("/result")
def show_result():
    # Get the predicted score from URL parameter
    score = request.args.get("score")
    risk = request.args.get("risk")
    return render_template("result.html", score=score, risk=risk)


@app.route('/loan_request', methods=['GET', 'POST'])
def loan_request():
    suggested_amount = None
    loan_status = None
    check_cccd = None

    if request.method == 'POST':
        action = request.form.get('action')
        print(f"Action: {action}")
        if action == 'request_loan':
            cccd = request.form.get('cccd')

            if not cccd:
                return 'Please provide both amount and CCCD.', 400

            # Giả sử bạn có một hàm để lấy thông tin người dùng từ CCCD
            user_info = get_user_info_by_cccd(cccd)
            if not user_info:
                return 'Invalid CCCD.', 400
            print("user info: ",user_info)
            # Tính toán điểm tín dụng
            input_data = {
                "LOAN": user_info['LOAN'],
                "MORTDUE": user_info['MORTDUE'],
                "VALUE": user_info['VALUE'],
                "REASON": user_info['REASON'],
                "JOB": user_info['JOB'],
                "YOJ": user_info['YOJ'],
                "DEROG": user_info['DEROG'],
                "DELINQ": user_info['DELINQ'],
                "CLAGE": user_info['CLAGE'],
                "NINQ": user_info['NINQ'],
                "CLNO": user_info['CLNO'],
                "DEBTINC": user_info['DEBTINC']
            }

            input_df = pd.DataFrame([input_data])
            input_df_temp = preProcessData(input_df)
            print("Processed DataFrame:", input_df_temp)
            pred_score = model.predict_proba(input_df_temp)[:, 1]
            print("Predicted Score Probability:", pred_score)
            score = credit_score(pred_score[0])
            print("Credit Score:", score)
            risk = 'high' if score <= 430 else 'medium' if score <= 679 else 'low'
            print("Risk Level:", risk)
            # Đề xuất số tiền có thể được vay
            suggested_amount = user_info['LOAN']
            if risk == 'medium':
                suggested_amount = user_info['LOAN'] * 0.9
            elif risk == 'high':
                suggested_amount = 0
                user_message = "Bạn không đủ điều kiện vay tiền."
            else:
                suggested_amount = user_info['LOAN']

            return render_template('loan_request.html', suggested_amount=suggested_amount, cccd=cccd, user_message=user_message if risk == 'high' else None)

        elif action == 'check_status':
            cccd = session.get('cccd')
            print(f"CCCD from session: {cccd}")
            if not cccd:
                print("No CCCD found in session.")
                return 'User not logged in or CCCD not found.', 403
            
            check_cccd = cccd
            print(f"Checking status for CCCD: {check_cccd}")

            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM status WHERE cccd = %s ORDER BY approval_date DESC LIMIT 1", (check_cccd,))
            loan_status = cursor.fetchone()
            connection.close()

            print(f"Loan Status: {loan_status}")

            return render_template('loan_request.html', loan_status=loan_status, check_cccd=check_cccd)

    return render_template('loan_request.html')

@app.route('/confirm_loan', methods=['POST'])
def confirm_loan():
    cccd = request.form.get('cccd')
    suggested_amount = float(request.form.get('suggested_amount'))

    connection = get_db_connection()
    cursor = connection.cursor()
    # Cập nhật bảng status với thông tin yêu cầu vay mới
    cursor.execute("""
        INSERT INTO status (cccd, amount, status, approval_date)
        VALUES (%s, %s, %s, %s)
    """, (cccd, suggested_amount, 'pending', datetime.now()))
    connection.commit()
    connection.close()

    return render_template('confirm_loan.html', cccd=cccd, suggested_amount=suggested_amount)

def get_user_info_by_cccd(cccd):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM loan_requests WHERE cccd = %s", (cccd,))
    user_info = cursor.fetchone()
    connection.close()
    return user_info

@app.route('/check_status', methods=['GET', 'POST'])
def check_status():
    if 'cccd' in session:
        cccd = session['cccd']
        if request.method == 'POST':
            cccd = request.form['cccd']
        
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM status WHERE cccd = %s ORDER BY id DESC LIMIT 1", (cccd,))
        status = cursor.fetchone()
        connection.close()

        return render_template('check_status.html', status=status)
    
    return redirect(url_for('login'))

def preProcessData(input_dataframe):
    # allowed columns
    column_names = [
        "LOAN",
        "MORTDUE",
        "VALUE",
        "REASON",
        "JOB",
        "YOJ",
        "DEROG",
        "DELINQ",
        "CLAGE",
        "NINQ",
        "CLNO",
        "DEBTINC",
    ]
    # creating temporary dataframe with allowed columns for processing
    modified_dataframe = input_dataframe[column_names].copy()

    median_columns = [
        "MORTDUE",
        "VALUE",
        "DEBTINC",
        "CLAGE",
        "YOJ",
        "CLNO",
        "NINQ",
        "DEROG",
        "DELINQ",
    ]
    mode_columns = ["REASON", "JOB"]

    for col in modified_dataframe:
        if col in median_columns:
            modified_dataframe[col] = modified_dataframe[col].fillna(
                modified_dataframe[col].median()
            )
        if col in mode_columns:
            modified_dataframe[col] = modified_dataframe[col].fillna(
                modified_dataframe[col].mode()
            )

    input_df_temp = pd.get_dummies(modified_dataframe, columns=["REASON", "JOB"])
    all_columns = [
        "LOAN",
        "MORTDUE",
        "VALUE",
        "YOJ",
        "DEROG",
        "DELINQ",
        "CLAGE",
        "NINQ",
        "CLNO",
        "DEBTINC",
        "REASON_DebtCon",
        "REASON_HomeImp",
        "JOB_Mgr",
        "JOB_Office",
        "JOB_Other",
        "JOB_ProfExe",
        "JOB_Sales",
        "JOB_Self",
    ]

    for col in all_columns:
        if col not in input_df_temp.columns:
            input_df_temp[col] = 0

    input_df_temp = input_df_temp[all_columns]

    input_df_temp = pd.DataFrame(input_df_temp, columns=all_columns)

    # Drop the 'BAD' column if it exists
    if "BAD" in input_df_temp.columns:
        input_df_temp = input_df_temp.drop("BAD", axis=1)

    input_df_temp = scaler.transform(input_df_temp)
    return input_df_temp


import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/nhanvien', methods=['GET', 'POST'])
def nhanvien():
    if 'role' in session and session['role'] == 'nhanvien':
        connection = get_db_connection()
        cursor = connection.cursor()

        # Lấy tất cả các yêu cầu vay đang chờ xử lý
        cursor.execute("SELECT * FROM status WHERE status = 'pending'")
        requests = cursor.fetchall()

        print("Pending Requests:", requests)

        loan_requests_details = []
        for req in requests:
            id, cccd, amount, status, approval_date = req

            # Lấy thông tin yêu cầu vay từ bảng loan_requests
            cursor.execute("SELECT * FROM loan_requests WHERE cccd = %s", (cccd,))
            loan_request = cursor.fetchone()

            print("Loan Request:", loan_request)  # In kết quả truy vấn để kiểm tra số lượng cột

            if loan_request:
                # Xử lý dữ liệu theo số lượng cột thực tế
                try:
                    req_id, cccd, name, phone, address, LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC = loan_request
                except ValueError:
                    print("Error unpacking loan_request:", loan_request)
                    continue
                # Tính toán điểm dựa trên thông tin yêu cầu vay
                input_data = {
                    "LOAN": LOAN,
                    "MORTDUE": MORTDUE,
                    "VALUE": VALUE,
                    "REASON": REASON,
                    "JOB": JOB,
                    "YOJ": YOJ,
                    "DEROG": DEROG,
                    "DELINQ": DELINQ,
                    "CLAGE": CLAGE,
                    "NINQ": NINQ,
                    "CLNO": CLNO,
                    "DEBTINC": DEBTINC
                }

                input_df = pd.DataFrame([input_data])
                input_df_temp = preProcessData(input_df)
                pred_score = model.predict_proba(input_df_temp)[:, 1]
                score = credit_score(pred_score[0])
                risk = credit_score_category(score)
                
                loan_requests_details.append({
                    'id': id,
                    'status': status,
                    'approval_date': approval_date,
                    'LOAN': LOAN,
                    'MORTDUE': MORTDUE,
                    'VALUE': VALUE,
                    'REASON': REASON,
                    'JOB': JOB,
                    'YOJ': YOJ,
                    'DEROG': DEROG,
                    'DELINQ': DELINQ,
                    'CLAGE': CLAGE,
                    'NINQ': NINQ,
                    'CLNO': CLNO,
                    'DEBTINC': DEBTINC,
                    'score': score,
                    'risk': risk,
                    'user': {
                        'cccd': cccd,
                        'name': name,
                        'phone': phone,
                        'address': address,
                    }
                })

        connection.close()
        return render_template('nhanvien.html', requests=loan_requests_details)
    
    return redirect(url_for('login'))


@app.route('/approve_request/<string:cccd>', methods=['POST'])
def approve_request(cccd):
    print(f"Attempting to approve request for CCCD: {cccd}")  # Debug statement

    if 'role' in session and session['role'] == 'nhanvien':
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("UPDATE status SET status = %s WHERE cccd = %s", ('approved',cccd,))
            connection.commit()
            print(f"Request approved for CCCD: {cccd}")  # Debug statement
        except Exception as e:
            print(f"Error approving request for CCCD: {cccd}: {e}")
        finally:
            connection.close()
        return redirect(url_for('nhanvien'))
    return redirect(url_for('login'))

@app.route('/reject_request/<string:cccd>', methods=['POST'])
def reject_request(cccd):
    if 'role' in session and session['role'] == 'nhanvien':
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("UPDATE status SET status = %s WHERE cccd = %s", ('rejected', cccd,))
            connection.commit()
            print(f"Request rejected for CCCD: {cccd}")  # Debug statement
        except Exception as e:
            print(f"Error rejecting request for CCCD: {cccd}: {e}")
        finally:
            connection.close()
        return redirect(url_for('nhanvien'))
    return redirect(url_for('login'))



@app.route('/update_loan_details', methods=['POST'])
def update_loan_details():
    if 'user_id' in session:
        user_id = session['user_id']
        data = {
            "LOAN": request.form['LOAN'],
            "MORTDUE": request.form['MORTDUE'],
            "VALUE": request.form['VALUE'],
            "REASON": request.form['REASON'],
            "JOB": request.form['JOB'],
            "YOJ": request.form['YOJ'],
            "DEROG": request.form['DEROG'],
            "DELINQ": request.form['DELINQ'],
            "CLAGE": request.form['CLAGE'],
            "NINQ": request.form['NINQ'],
            "CLNO": request.form['CLNO'],
            "DEBTINC": request.form['DEBTINC'],
        }

        connection = get_db_connection()
        cursor = connection.cursor()

        # Update loan details in users table
        cursor.execute("""
            UPDATE users SET
                LOAN = %s, MORTDUE = %s, VALUE = %s, REASON = %s, JOB = %s,
                YOJ = %s, DEROG = %s, DELINQ = %s, CLAGE = %s, NINQ = %s,
                CLNO = %s, DEBTINC = %s
            WHERE id = %s
        """, (
            data["LOAN"], data["MORTDUE"], data["VALUE"], data["REASON"],
            data["JOB"], data["YOJ"], data["DEROG"], data["DELINQ"], data["CLAGE"],
            data["NINQ"], data["CLNO"], data["DEBTINC"], user_id
        ))

        connection.commit()
        connection.close()

        return redirect(url_for('profile'))
    return redirect(url_for('home'))

# Load the model
with open("D:/FSS/credit_score/model/XGBClassifier2.pkl", "rb") as file:
    model = pickle.load(file)

with open("D:/FSS/credit_score/standard/StandardScaler2.pkl", "rb") as file:
    scaler = pickle.load(file)


def credit_score(p):
    factor = 25 / np.log(2)
    offset = 600 - factor * np.log(50)
    val = (1 - p) / p
    score = offset + factor * np.log(val)
    return round(score)

def credit_score_category(score):
    if score < 580:
        return "High"
    elif score < 670:
        return "Medium-High"
    elif score < 740:
        return "Medium"
    elif score < 800:
        return "Medium-Low"
    else:
        return "Low"


if __name__ == '__main__':
    app.run(debug=True)
