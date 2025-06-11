"""
Synthetic data generator for Social Support AI Workflow

Generates realistic test data for:
- Social support applications
- Bank statements
- Emirates ID information
- Resume data
- Credit reports
- Assets/liabilities data
"""
import random
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from faker import Faker
import uuid

fake = Faker()


class SyntheticDataGenerator:
    """Generates synthetic data for testing the social support application system"""
    
    def __init__(self):
        self.emirates = ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"]
        self.employment_statuses = ["employed", "unemployed", "self_employed", "student", "retired"]
        self.income_ranges = {
            "low": (1000, 3000),
            "medium": (3000, 8000),
            "high": (8000, 15000),
            "very_high": (15000, 30000)
        }
    
    def generate_application_data(self, count: int = 1) -> List[Dict[str, Any]]:
        """Generate synthetic social support application data"""
        applications = []
        
        for _ in range(count):
            # Determine income category to influence other factors
            income_category = random.choices(
                ["low", "medium", "high", "very_high"],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]
            
            monthly_income = random.randint(*self.income_ranges[income_category])
            
            # Family size influences support likelihood
            family_size = random.randint(1, 8)
            
            application = {
                "application_id": f"APP-{uuid.uuid4().hex[:8].upper()}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.email(),
                "phone": fake.phone_number(),
                "emirates_id": self._generate_emirates_id(),
                "address": fake.address(),
                "city": fake.city(),
                "emirate": random.choice(self.emirates),
                "monthly_income": monthly_income,
                "employment_status": random.choice(self.employment_statuses),
                "family_size": family_size,
                "created_at": fake.date_time_between(start_date="-6M", end_date="now"),
                
                # Additional fields for testing
                "has_savings": random.choice([True, False]),
                "savings_amount": random.randint(0, 50000) if random.random() > 0.3 else 0,
                "has_debts": random.choice([True, False]),
                "debt_amount": random.randint(0, 100000) if random.random() > 0.4 else 0,
                "education_level": random.choice([
                    "no_education", "primary", "secondary", "diploma", "bachelor", "master", "phd"
                ]),
                "employment_duration_months": random.randint(0, 120),
                "previous_applications": random.randint(0, 3),
                
                # Risk factors
                "has_criminal_record": random.random() < 0.05,  # 5% chance
                "credit_score": random.randint(300, 850),
                
                # Need indicators
                "housing_type": random.choice(["owned", "rented", "family_house", "shared"]),
                "monthly_rent": random.randint(500, 8000) if random.random() > 0.3 else 0,
                "has_medical_conditions": random.random() < 0.15,  # 15% chance
                "number_of_dependents": max(0, family_size - 2),
            }
            
            applications.append(application)
        
        return applications
    
    def generate_bank_statement_data(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic bank statement data based on application"""
        monthly_income = application_data["monthly_income"]
        employment_status = application_data["employment_status"]
        
        # Generate transactions for last 3 months
        transactions = []
        start_date = datetime.now() - timedelta(days=90)
        
        for i in range(90):
            current_date = start_date + timedelta(days=i)
            
            # Add salary if employed (monthly)
            if employment_status == "employed" and current_date.day <= 5:
                transactions.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "description": "Salary Credit",
                    "amount": monthly_income + random.randint(-200, 200),
                    "type": "credit",
                    "balance": random.randint(1000, 10000)
                })
            
            # Add random expenses
            if random.random() < 0.7:  # 70% chance of expense per day
                expense_amount = random.randint(10, 500)
                transactions.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "description": random.choice([
                        "Grocery Shopping", "Fuel Purchase", "Restaurant",
                        "Utility Bill", "Phone Bill", "Online Purchase",
                        "ATM Withdrawal", "Medical Expense"
                    ]),
                    "amount": -expense_amount,
                    "type": "debit",
                    "balance": random.randint(100, 5000)
                })
        
        # Calculate summary statistics
        credits = [t["amount"] for t in transactions if t["amount"] > 0]
        debits = [abs(t["amount"]) for t in transactions if t["amount"] < 0]
        
        return {
            "account_holder": f"{application_data['first_name']} {application_data['last_name']}",
            "account_number": f"****{random.randint(1000, 9999)}",
            "statement_period": "Last 3 months",
            "transactions": transactions,
            "summary": {
                "total_credits": sum(credits),
                "total_debits": sum(debits),
                "average_monthly_income": sum(credits) / 3 if credits else 0,
                "average_monthly_expenses": sum(debits) / 3 if debits else 0,
                "transaction_count": len(transactions)
            }
        }
    
    def generate_emirates_id_data(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic Emirates ID data"""
        return {
            "emirates_id": application_data["emirates_id"],
            "name_arabic": fake.name(),  # Would be Arabic in real scenario
            "name_english": f"{application_data['first_name']} {application_data['last_name']}",
            "nationality": "UAE" if random.random() > 0.3 else fake.country(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d"),
            "gender": random.choice(["Male", "Female"]),
            "address": application_data["address"],
            "emirate": application_data["emirate"],
            "issue_date": fake.date_between(start_date="-10y", end_date="now").strftime("%Y-%m-%d"),
            "expiry_date": fake.date_between(start_date="now", end_date="+5y").strftime("%Y-%m-%d"),
            "card_status": "Active"
        }
    
    def generate_resume_data(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic resume data"""
        education_level = application_data["education_level"]
        employment_duration = application_data["employment_duration_months"]
        
        # Generate education history
        education = []
        if education_level != "no_education":
            education.append({
                "degree": education_level,
                "institution": fake.company(),
                "year": random.randint(2000, 2022),
                "field": random.choice([
                    "Business Administration", "Engineering", "Computer Science",
                    "Medicine", "Education", "Arts", "Science"
                ])
            })
        
        # Generate work experience
        experience = []
        if employment_duration > 0:
            num_jobs = min(5, max(1, employment_duration // 24))  # One job per 2 years average
            
            for i in range(num_jobs):
                start_date = fake.date_between(start_date="-10y", end_date="-1y")
                end_date = fake.date_between(start_date=start_date, end_date="now")
                
                experience.append({
                    "position": fake.job(),
                    "company": fake.company(),
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "duration_months": (end_date - start_date).days // 30,
                    "responsibilities": [
                        fake.sentence() for _ in range(random.randint(2, 5))
                    ]
                })
        
        return {
            "personal_info": {
                "name": f"{application_data['first_name']} {application_data['last_name']}",
                "email": application_data["email"],
                "phone": application_data["phone"],
                "address": application_data["address"]
            },
            "education": education,
            "experience": experience,
            "skills": [fake.word() for _ in range(random.randint(3, 10))],
            "languages": random.sample(["English", "Arabic", "Hindi", "Urdu", "French"], 
                                     random.randint(1, 3))
        }
    
    def generate_credit_report_data(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic credit report data"""
        credit_score = application_data["credit_score"]
        has_debts = application_data["has_debts"]
        debt_amount = application_data["debt_amount"]
        
        # Generate credit accounts
        accounts = []
        if has_debts and debt_amount > 0:
            num_accounts = random.randint(1, 5)
            remaining_debt = debt_amount
            
            for i in range(num_accounts):
                account_debt = random.randint(100, min(remaining_debt, 50000))
                remaining_debt -= account_debt
                
                accounts.append({
                    "account_type": random.choice([
                        "Credit Card", "Personal Loan", "Car Loan", 
                        "Home Loan", "Business Loan"
                    ]),
                    "bank_name": fake.company(),
                    "balance": account_debt,
                    "credit_limit": account_debt * random.uniform(1.2, 3.0),
                    "payment_history": random.choice(["Good", "Fair", "Poor"]),
                    "account_status": "Active",
                    "opened_date": fake.date_between(start_date="-5y", end_date="now").strftime("%Y-%m-%d")
                })
        
        return {
            "personal_info": {
                "name": f"{application_data['first_name']} {application_data['last_name']}",
                "emirates_id": application_data["emirates_id"],
                "report_date": datetime.now().strftime("%Y-%m-%d")
            },
            "credit_score": {
                "score": credit_score,
                "rating": self._get_credit_rating(credit_score),
                "factors": [
                    "Payment history", "Credit utilization", "Length of credit history"
                ]
            },
            "accounts": accounts,
            "summary": {
                "total_accounts": len(accounts),
                "total_debt": sum(acc["balance"] for acc in accounts),
                "total_credit_limit": sum(acc["credit_limit"] for acc in accounts),
                "credit_utilization": (sum(acc["balance"] for acc in accounts) / 
                                     max(1, sum(acc["credit_limit"] for acc in accounts))) * 100
            }
        }
    
    def generate_assets_liabilities_data(self, application_data: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic assets and liabilities data as Excel-ready DataFrame"""
        has_savings = application_data["has_savings"]
        savings_amount = application_data["savings_amount"]
        has_debts = application_data["has_debts"]
        debt_amount = application_data["debt_amount"]
        
        data = []
        
        # Assets
        if has_savings and savings_amount > 0:
            data.append({
                "Category": "Assets",
                "Type": "Bank Savings",
                "Description": "Savings Account",
                "Value": savings_amount,
                "Currency": "AED"
            })
        
        # Add some random assets
        if random.random() > 0.7:  # 30% chance of having a car
            data.append({
                "Category": "Assets",
                "Type": "Vehicle",
                "Description": fake.word() + " Car",
                "Value": random.randint(20000, 150000),
                "Currency": "AED"
            })
        
        if random.random() > 0.8:  # 20% chance of property
            data.append({
                "Category": "Assets",
                "Type": "Real Estate",
                "Description": "Residential Property",
                "Value": random.randint(500000, 2000000),
                "Currency": "AED"
            })
        
        # Liabilities
        if has_debts and debt_amount > 0:
            # Distribute debt across different types
            remaining_debt = debt_amount
            debt_types = ["Credit Card", "Personal Loan", "Car Loan"]
            
            for debt_type in debt_types:
                if remaining_debt <= 0:
                    break
                
                if random.random() > 0.5:  # 50% chance for each debt type
                    amount = min(remaining_debt, random.randint(5000, 50000))
                    remaining_debt -= amount
                    
                    data.append({
                        "Category": "Liabilities",
                        "Type": debt_type,
                        "Description": f"{debt_type} Debt",
                        "Value": amount,
                        "Currency": "AED"
                    })
        
        return pd.DataFrame(data)
    
    def _generate_emirates_id(self) -> str:
        """Generate a synthetic Emirates ID number"""
        # Format: XXX-XXXX-XXXXXXX-X
        part1 = random.randint(100, 999)
        part2 = random.randint(1000, 9999)
        part3 = random.randint(1000000, 9999999)
        part4 = random.randint(1, 9)
        
        return f"{part1}-{part2}-{part3}-{part4}"
    
    def _get_credit_rating(self, score: int) -> str:
        """Convert credit score to rating"""
        if score >= 750:
            return "Excellent"
        elif score >= 700:
            return "Good"
        elif score >= 650:
            return "Fair"
        elif score >= 600:
            return "Poor"
        else:
            return "Very Poor"
    
    def save_synthetic_dataset(self, count: int = 50, output_dir: str = "./data/synthetic"):
        """Generate and save a complete synthetic dataset"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        applications = self.generate_application_data(count)
        
        # Save applications
        applications_df = pd.DataFrame(applications)
        applications_df.to_csv(f"{output_dir}/applications.csv", index=False)
        
        # Generate related documents for each application
        for i, app in enumerate(applications):
            app_id = app["application_id"]
            
            # Bank statement
            bank_data = self.generate_bank_statement_data(app)
            with open(f"{output_dir}/bank_statement_{app_id}.json", "w") as f:
                json.dump(bank_data, f, indent=2)
            
            # Emirates ID
            emirates_data = self.generate_emirates_id_data(app)
            with open(f"{output_dir}/emirates_id_{app_id}.json", "w") as f:
                json.dump(emirates_data, f, indent=2)
            
            # Resume
            resume_data = self.generate_resume_data(app)
            with open(f"{output_dir}/resume_{app_id}.json", "w") as f:
                json.dump(resume_data, f, indent=2)
            
            # Credit report
            credit_data = self.generate_credit_report_data(app)
            with open(f"{output_dir}/credit_report_{app_id}.json", "w") as f:
                json.dump(credit_data, f, indent=2)
            
            # Assets/Liabilities
            assets_df = self.generate_assets_liabilities_data(app)
            assets_df.to_excel(f"{output_dir}/assets_liabilities_{app_id}.xlsx", index=False)
        
        print(f"Generated synthetic dataset with {count} applications in {output_dir}")
        return applications 