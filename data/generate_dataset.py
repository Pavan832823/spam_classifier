"""
Dataset Generator for Email Spam Classifier
Generates a realistic synthetic dataset when Enron/SMS datasets are not available.
In production, replace with: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
"""

import pandas as pd
import random
import os

random.seed(42)

# ─── Ham (Legitimate) email templates ───────────────────────────────────────
HAM_SUBJECTS = [
    "Meeting at 3pm today",
    "Re: Project update",
    "Lunch tomorrow?",
    "Your invoice is ready",
    "Team standup notes",
    "Happy Birthday!",
    "Flight confirmation",
    "Question about the report",
    "Weekend plans",
    "Follow-up from our call",
    "Feedback on your presentation",
    "Welcome to the team",
    "Your order has shipped",
    "Interview schedule",
    "Reminder: dentist appointment",
]

HAM_BODIES = [
    "Hi, just wanted to confirm our meeting scheduled for 3pm today in conference room B. Please bring the Q3 reports.",
    "Following up on our project discussion from last week. The deadline has been moved to Friday. Let me know if that works for you.",
    "Are you free for lunch tomorrow around noon? I know a great Italian place near the office.",
    "Your invoice #1042 for last month's services is attached. Please review and process at your earliest convenience.",
    "Notes from today's standup: Alice is working on the API, Bob is fixing the login bug, Charlie will handle deployment.",
    "Just wanted to wish you a very happy birthday! Hope you have a wonderful day with family and friends.",
    "Your flight booking is confirmed. Flight AI-202 departs at 6:45 AM from Terminal 2. Check-in opens 3 hours before departure.",
    "I had a few questions about the quarterly report you sent. Can we schedule a quick call to go over section 3?",
    "Hey, are you guys doing anything this weekend? We're thinking of going hiking if the weather holds up.",
    "Great talking to you earlier! As discussed, I'll send over the proposal by end of day Thursday.",
    "Your presentation yesterday was excellent. The stakeholders were very impressed with the data visualization.",
    "Welcome aboard! We're excited to have you on the team. Your onboarding schedule is attached.",
    "Good news! Your order #ORD-5893 has shipped and will arrive by Thursday. Tracking number: TRK29384756.",
    "Your interview is scheduled for Monday at 2:00 PM. Please bring a copy of your resume and portfolio.",
    "Reminder: You have a dentist appointment tomorrow at 10:30 AM with Dr. Sharma. Please arrive 10 minutes early.",
    "Could you please review the attached document and send me your comments by Friday? Thanks!",
    "The quarterly sales report shows a 12% increase over last quarter. Great work from everyone on the sales team.",
    "I'm forwarding you the email chain about the client request. Please get in touch with them directly.",
    "The server maintenance is scheduled for Sunday at 2 AM. Expected downtime is 2 hours. Please plan accordingly.",
    "Thank you for your application. We'd like to invite you for a technical interview next week.",
]

# ─── Spam email templates ────────────────────────────────────────────────────
SPAM_SUBJECTS = [
    "YOU HAVE WON $1,000,000!!!",
    "Urgent: Your account needs verification",
    "Make money fast from home!",
    "Limited time offer - Act NOW",
    "Your PayPal account is suspended",
    "Congratulations! You are selected",
    "FREE iPhone 15 - Claim Now!",
    "URGENT: Update your banking info",
    "Work from home - $5000/week GUARANTEED",
    "Hot singles in your area",
    "Nigerian Prince needs your help",
    "Lose 30 pounds in 30 days",
    "Click here to claim your prize",
    "Your subscription expires today",
    "FINAL NOTICE: Payment required",
]

SPAM_BODIES = [
    "CONGRATULATIONS! You have been selected as our lucky winner for $1,000,000 USD lottery. Send your bank details immediately to claim your prize. ACT NOW before it expires!!!",
    "Your account has been SUSPENDED due to suspicious activity. Click here immediately to verify your identity or your account will be PERMANENTLY DELETED: http://verify-account-now.xyz",
    "MAKE $5000 A WEEK FROM HOME! No experience needed. Thousands already making money! Click here to start TODAY. Limited spots available!!!",
    "AMAZING LIMITED TIME OFFER! Buy one get three FREE! Today only! This deal expires in 2 hours! Click now! Don't miss out! Best price ever!!!",
    "Dear Customer, Your PayPal account has been limited. You must update your information within 24 hours or your account will be suspended. Login here: http://paypal-secure-verify.ru",
    "You have been selected from millions of participants to receive a special gift worth $500. Claim your prize now by providing your personal details to our representative.",
    "FREE iPhone 15 Pro giveaway! You've been chosen! Just pay $2.99 shipping and handling. Click link to claim: http://free-iphone-claim.biz. Offer expires tonight!",
    "URGENT BANK NOTICE: We have detected unauthorized access to your account. Verify your identity NOW by entering your card details at our secure portal.",
    "Are you tired of your 9-5 job? Make $10,000 per month working just 2 hours a day from home! Our proven system guarantees results! Click here to join THOUSANDS of success stories!",
    "Hot singles are waiting to meet you in your area! Join FREE today and find your perfect match! Thousands of beautiful women waiting! Click here NOW!",
    "CONFIDENTIAL BUSINESS PROPOSAL: I am Prince Emmanuel Adeyemi of Nigeria. I have $45 million that I need to transfer out of the country. I need your help and will pay you 30% commission.",
    "SHOCKING weight loss secret doctors don't want you to know! Lose 30 pounds in 30 days with this one weird trick! 100% natural, no exercise needed! Order now and get 60% OFF!",
    "YOU HAVE WON! Your email address was selected in our monthly draw. Click the link below to claim your $500 Amazon gift card. Hurry, link expires in 24 hours!",
    "Your Netflix subscription has EXPIRED. Your account will be DELETED unless you renew immediately. Click here to update payment: http://netflix-billing-update.net",
    "FINAL NOTICE: This is your last chance to pay your outstanding debt of $2,847. Failure to respond will result in LEGAL ACTION. Contact us immediately to avoid consequences.",
    "Congratulations on your selection for our VIP program! You qualify for a $1000 loan with no credit check! Instant approval! Reply with your SSN to confirm eligibility.",
    "PHARMACY SPECIAL: Cheap meds without prescription! Viagra, Cialis, and more! Discreet delivery worldwide! Best prices on the internet! Order now get 70% discount!!!",
    "Your computer has been INFECTED with a virus! Call our toll-free number 1-800-SCAM-NOW immediately! Microsoft certified technicians will fix it remotely for only $299!",
    "EARN BITCOIN FAST! Our automated trading bot guarantees 500% returns daily! Zero risk! Just deposit $100 to start and watch the money roll in! Join 50,000 satisfied members!",
    "Dear Friend, I found your email on the internet and I need a trustworthy person to help me claim my late husband's inheritance of $12.5 million. I will give you 40% as your share.",
]


def generate_dataset(n_samples: int = 1000, output_path: str = "emails.csv") -> pd.DataFrame:
    """Generate a balanced synthetic email dataset."""
    
    ham_count = int(n_samples * 0.6)   # 60% ham (realistic ratio)
    spam_count = n_samples - ham_count  # 40% spam

    records = []

    # Generate ham emails
    for i in range(ham_count):
        subject = random.choice(HAM_SUBJECTS)
        body = random.choice(HAM_BODIES)
        # Add some variation
        senders = ["john.doe@company.com", "hr@office.org", "team@startup.io",
                   "noreply@airline.com", "billing@service.com", "friend@gmail.com"]
        records.append({
            "sender": random.choice(senders),
            "subject": subject,
            "body": body,
            "text": f"Subject: {subject}\n\n{body}",
            "label": 0,  # ham
            "label_name": "ham"
        })

    # Generate spam emails
    for i in range(spam_count):
        subject = random.choice(SPAM_SUBJECTS)
        body = random.choice(SPAM_BODIES)
        senders = ["prize@winner-notify.com", "noreply@verify-account.xyz",
                   "offer@deals-today.biz", "support@paypa1-secure.ru",
                   "admin@free-money.net", "info@lottery-winner.co"]
        records.append({
            "sender": random.choice(senders),
            "subject": subject,
            "body": body,
            "text": f"Subject: {subject}\n\n{body}",
            "label": 1,  # spam
            "label_name": "spam"
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset generated: {len(df)} emails")
    print(f"   Ham:  {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"   Spam: {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"   Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    generate_dataset(2000, "emails.csv")
