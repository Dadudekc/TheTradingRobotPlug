"""
gui/notifications.py
--------------------
Defines a simple email notification function.
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_notification(subject, message,
                      smtp_server="smtp.example.com",
                      smtp_port=587,
                      sender_email="sender@example.com",
                      sender_password="password",
                      recipient_email="recipient@example.com"):
    """
    Sends an email notification with the given subject and message.
    
    Args:
        subject (str): The email subject.
        message (str): The email body.
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port.
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password.
        recipient_email (str): Recipient's email address.
    """
    try:
        # Create a secure SSL context
        context = ssl.create_default_context()
        
        # Create the email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        
        # Establish connection and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print("Notification sent successfully.")
    except Exception as e:
        print(f"Failed to send notification: {str(e)}")
