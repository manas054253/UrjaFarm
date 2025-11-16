import os
import joblib
import pandas as pd
import plotly
import plotly.graph_objects as go
import json
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# ===== 1. APP & DB SETUP =====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_change_this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost/credit_app_db'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

# ===== 2. LOAD ML MODEL =====
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'credit_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

MODEL_FEATURES = [
    'bills_paid_on_time_12m',
    'mobile_age_years',
    'avg_daily_sales',
    'avg_monthly_recharge',
    'avg_bill_amount',
    'is_bill_overdue_30d',
    'days_with_no_sales'
]

# ===== 3. DATABASE MODELS =====
class Officer(UserMixin, db.Model):
    """Officer/Admin user table"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Applicant(UserMixin, db.Model):
    """Applicant/MSME user table"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    applications = db.relationship('Application', backref='applicant', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Application(db.Model):
    """Loan application table with FEEDBACK LEARNING columns"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    applicant_id = db.Column(db.Integer, db.ForeignKey('applicant.id'), nullable=False)
    
    # Application data
    bills_paid_on_time_12m = db.Column(db.Integer)
    avg_bill_amount = db.Column(db.Float)
    is_bill_overdue_30d = db.Column(db.Integer)
    mobile_age_years = db.Column(db.Integer)
    avg_monthly_recharge = db.Column(db.Float)
    avg_daily_sales = db.Column(db.Float)
    days_with_no_sales = db.Column(db.Integer)
    
    # AI Model Results
    credit_score = db.Column(db.Integer)
    risk_status = db.Column(db.String(100))
    recommendation = db.Column(db.String(500))
    
    # FEEDBACK LEARNING COLUMNS
    status = db.Column(db.String(50), default='Pending')
    officer_feedback = db.Column(db.Text, nullable=True)
    officer_id = db.Column(db.Integer, db.ForeignKey('officer.id'), nullable=True)
    decision_timestamp = db.Column(db.DateTime, nullable=True)
    user_response = db.Column(db.Text, nullable=True)
    response_timestamp = db.Column(db.DateTime, nullable=True)
    submission_count = db.Column(db.Integer, default=1)
    
    officer = db.relationship('Officer')
    actions = db.relationship('ApplicationAction', backref='application')


class ApplicationAction(db.Model):
    """Audit trail - logs all application actions"""
    id = db.Column(db.Integer, primary_key=True)
    application_id = db.Column(db.Integer, db.ForeignKey('application.id'), nullable=False)
    officer_id = db.Column(db.Integer, db.ForeignKey('officer.id'), nullable=True)
    action_type = db.Column(db.String(50))
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    
    officer = db.relationship('Officer')


# ===== 4. FLASK-LOGIN USER LOADER =====
@login_manager.user_loader
def load_user(user_id):
    return Officer.query.get(int(user_id))


# ===== 5. PUBLIC ROUTES =====

@app.route('/')
def home():
    """Landing page"""
    return render_template('landing_page.html')


# ===== 6. APPLICANT ROUTES =====

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Applicant registration"""
    if 'applicant_id' in session:
        return redirect(url_for('applicant_dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        applicant = Applicant.query.filter_by(email=email).first()
        if applicant:
            flash('Email address already registered.', 'danger')
            return redirect(url_for('register'))

        new_applicant = Applicant(name=name, email=email)
        new_applicant.set_password(password)
        db.session.add(new_applicant)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Applicant login"""
    if 'applicant_id' in session:
        return redirect(url_for('applicant_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        applicant = Applicant.query.filter_by(email=email).first()
        if applicant and applicant.check_password(password):
            session['applicant_id'] = applicant.id
            session['applicant_name'] = applicant.name
            flash('Logged in successfully.', 'success')
            return redirect(url_for('applicant_dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/applicant_logout')
def applicant_logout():
    """Applicant logout"""
    session.pop('applicant_id', None)
    session.pop('applicant_name', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))


@app.route('/dashboard')
def applicant_dashboard():
    """User dashboard - shows form OR status"""
    if 'applicant_id' not in session:
        flash('You must be logged in.', 'warning')
        return redirect(url_for('login'))

    applicant_id = session['applicant_id']
    existing_app = Application.query.filter_by(applicant_id=applicant_id).first()

    if existing_app:
        return render_template('applicant_dashboard.html', app=existing_app, mode='status')
    else:
        return render_template('applicant_dashboard.html', mode='form')


@app.route('/apply')
def apply():
    """Redirect to dashboard"""
    if 'applicant_id' not in session:
        flash('You must be logged in to access this page.', 'warning')
        return redirect(url_for('login'))
    return redirect(url_for('applicant_dashboard'))


@app.route('/predict', methods=['POST'])
def predict():
    """Process application form"""
    if 'applicant_id' not in session:
        flash('Your session expired. Please log in again.', 'warning')
        return redirect(url_for('login'))

    if model is None:
        flash("Model is not loaded. Cannot make predictions.", "danger")
        return redirect(url_for('applicant_dashboard'))

    try:
        form_data = {
            'bills_paid_on_time_12m': request.form.get('bills_paid_on_time_12m', type=int),
            'avg_bill_amount': request.form.get('avg_bill_amount', type=float),
            'is_bill_overdue_30d': request.form.get('is_bill_overdue_30d', type=int),
            'mobile_age_years': request.form.get('mobile_age_years', type=int),
            'avg_monthly_recharge': request.form.get('avg_monthly_recharge', type=float),
            'avg_daily_sales': request.form.get('avg_daily_sales', type=float),
            'days_with_no_sales': request.form.get('days_with_no_sales', type=int)
        }

        input_df = pd.DataFrame([form_data], columns=MODEL_FEATURES)
        probabilities = model.predict_proba(input_df)
        prob_no_default = probabilities[0][0]
        score = int(prob_no_default * 1000)

        if score > 700:
            risk_status = "Low Risk"
            recommendation = "Recommend approval. Loan Amount: High. Interest Rate: Low (14-16%)."
        elif score > 450:
            risk_status = "Medium Risk"
            recommendation = "Recommend manual review. Loan Amount: Medium. Interest Rate: Medium (19-22%)."
        else:
            risk_status = "High Risk"
            recommendation = "Recommend rejection. Loan Amount: Low/None. Interest Rate: High (25%+)."

        new_application = Application(
            applicant_id=session['applicant_id'],
            credit_score=score,
            risk_status=risk_status,
            recommendation=recommendation,
            status='Pending',
            **form_data
        )

        db.session.add(new_application)
        db.session.flush()

        action = ApplicationAction(
            application_id=new_application.id,
            action_type='submitted',
            comment='Application submitted by user'
        )

        db.session.add(action)
        db.session.commit()

        flash('Your application has been submitted successfully and is under review.', 'success')
        return redirect(url_for('application_submitted'))

    except Exception as e:
        print(f"Error during prediction: {e}")
        flash(f"An error occurred: {e}", "danger")
        return redirect(url_for('applicant_dashboard'))


@app.route('/application_submitted')
def application_submitted():
    """Thank you page after submission"""
    if 'applicant_id' not in session:
        return redirect(url_for('login'))
    return render_template('application_submitted.html')


@app.route('/submit_user_response/<int:app_id>', methods=['POST'])
def submit_user_response(app_id):
    """User responds to 'More Info Requested'"""
    if 'applicant_id' not in session:
        flash('Session expired. Please log in again.', 'warning')
        return redirect(url_for('login'))

    try:
        app = Application.query.get_or_404(app_id)

        if app.applicant_id != session['applicant_id']:
            flash('Unauthorized access.', 'danger')
            return redirect(url_for('applicant_dashboard'))

        if app.status != 'More Info Requested':
            flash('This application is not awaiting your response.', 'warning')
            return redirect(url_for('applicant_dashboard'))

        user_response = request.form.get('user_response', '').strip()

        if not user_response:
            flash('Please provide a response.', 'warning')
            return redirect(url_for('applicant_dashboard'))

        app.user_response = user_response
        app.response_timestamp = datetime.utcnow()
        app.status = 'Pending'
        app.submission_count += 1

        action = ApplicationAction(
            application_id=app_id,
            action_type='user_responded',
            comment=f"User resubmitted with response: {user_response[:100]}..."
        )

        db.session.add(action)
        db.session.commit()

        flash('Your response has been submitted. Officer will review shortly.', 'success')
        return redirect(url_for('applicant_dashboard'))

    except Exception as e:
        print(f"Error in submit_user_response: {e}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('applicant_dashboard'))


# ===== 7. OFFICER ROUTES =====

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    """Officer login"""
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = Officer.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('admin_login.html')


@app.route('/logout')
@login_required
def logout():
    """Officer logout"""
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    """Officer dashboard - all applications"""
    try:
        all_applications = Application.query.order_by(Application.timestamp.desc()).all()
    except Exception as e:
        print(f"Error querying applications: {e}")
        flash("Could not load applications.", "danger")
        all_applications = []

    return render_template('admin_dashboard.html', applications=all_applications)


@app.route('/view/<int:app_id>')
@login_required
def view_application(app_id):
    """Officer views detailed application"""
    try:
        application = Application.query.get_or_404(app_id)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=application.credit_score,
            title={'text': f"Score: {application.risk_status}", 'font': {'size': 20, 'family': 'Inter'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 450], 'color': '#fecaca'},
                    {'range': [450, 700], 'color': '#fef08a'},
                    {'range': [700, 1000], 'color': '#bbf7d0'}
                ],
                'threshold': {
                    'line': {'color': "#4b5563", 'width': 4},
                    'thickness': 0.9,
                    'value': application.credit_score
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Inter, sans-serif", 'color': "#111827"},
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('view_application_with_feedback.html', app=application, chart_json=chart_json)

    except Exception as e:
        print(f"Error fetching application {app_id}: {e}")
        flash(f"Could not load the requested application: {e}", "danger")
        return redirect(url_for('admin_dashboard'))


@app.route('/approve_application/<int:app_id>', methods=['POST'])
@login_required
def approve_application(app_id):
    """Officer approves application"""
    try:
        app = Application.query.get_or_404(app_id)
        feedback = request.form.get('feedback', '').strip()

        if not feedback:
            flash('Feedback is required to make a decision.', 'warning')
            return redirect(url_for('view_application', app_id=app_id))

        app.status = 'Accepted'
        app.officer_id = current_user.id
        app.officer_feedback = feedback
        app.decision_timestamp = datetime.utcnow()

        action = ApplicationAction(
            application_id=app_id,
            officer_id=current_user.id,
            action_type='approved',
            comment=feedback
        )

        db.session.add(action)
        db.session.commit()

        flash('Application approved!', 'success')
        return redirect(url_for('admin_dashboard'))

    except Exception as e:
        print(f"Error approving: {e}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('view_application', app_id=app_id))


@app.route('/reject_application/<int:app_id>', methods=['POST'])
@login_required
def reject_application(app_id):
    """Officer rejects application"""
    try:
        app = Application.query.get_or_404(app_id)
        feedback = request.form.get('feedback', '').strip()

        if not feedback:
            flash('Feedback (reason for rejection) is required.', 'warning')
            return redirect(url_for('view_application', app_id=app_id))

        app.status = 'Rejected'
        app.officer_id = current_user.id
        app.officer_feedback = feedback
        app.decision_timestamp = datetime.utcnow()

        action = ApplicationAction(
            application_id=app_id,
            officer_id=current_user.id,
            action_type='rejected',
            comment=feedback
        )

        db.session.add(action)
        db.session.commit()

        flash('Application rejected.', 'info')
        return redirect(url_for('admin_dashboard'))

    except Exception as e:
        print(f"Error rejecting: {e}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('view_application', app_id=app_id))


@app.route('/request_more_info/<int:app_id>', methods=['POST'])
@login_required
def request_more_info(app_id):
    """Officer requests more info"""
    try:
        app = Application.query.get_or_404(app_id)
        feedback = request.form.get('feedback', '').strip()

        if not feedback:
            feedback = "Please provide additional information to support your application."

        app.status = 'More Info Requested'
        app.officer_id = current_user.id
        app.officer_feedback = feedback
        app.decision_timestamp = datetime.utcnow()

        action = ApplicationAction(
            application_id=app_id,
            officer_id=current_user.id,
            action_type='more_info_requested',
            comment=feedback
        )

        db.session.add(action)
        db.session.commit()

        flash('Request for more info sent to applicant.', 'success')
        return redirect(url_for('admin_dashboard'))

    except Exception as e:
        print(f"Error requesting info: {e}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('view_application', app_id=app_id))


@app.route('/officer_analytics')
@login_required
def officer_analytics():
    """Analytics dashboard"""
    try:
        all_apps = Application.query.filter(Application.status.in_(['Accepted', 'Rejected', 'More Info Requested'])).all()

        total_decisions = len(all_apps)
        approved_count = len([a for a in all_apps if a.status == 'Accepted'])
        rejected_count = len([a for a in all_apps if a.status == 'Rejected'])
        more_info_count = len([a for a in all_apps if a.status == 'More Info Requested'])

        approval_rate = (approved_count / total_decisions * 100) if total_decisions > 0 else 0

        rejection_reasons = {}
        for app in all_apps:
            if app.status == 'Rejected' and app.officer_feedback:
                reason = app.officer_feedback[:50]
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        top_reasons = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]

        stats = {
            'total_decisions': total_decisions,
            'approved': approved_count,
            'rejected': rejected_count,
            'more_info': more_info_count,
            'approval_rate': round(approval_rate, 2),
            'top_rejection_reasons': top_reasons
        }

        return render_template('officer_analytics.html', stats=stats)

    except Exception as e:
        print(f"Error in analytics: {e}")
        flash(f'Error loading analytics: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))


# ===== 8. START APP =====
if __name__ == '__main__':
    with app.app_context():
        # Create all tables
        db.create_all()

        # Create default admin user
        if not Officer.query.filter_by(username='admin').first():
            print("Creating default admin user...")
            default_user = Officer(username='admin')
            default_user.set_password('admin123')
            db.session.add(default_user)
            db.session.commit()
            print("✅ User 'admin' with password 'admin123' created.")
        else:
            print("✅ Admin user already exists.")

    print("✅ All tables created successfully!")
    print("🚀 Starting Flask app...")
    app.run(debug=True)