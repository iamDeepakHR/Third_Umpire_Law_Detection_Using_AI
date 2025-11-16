#!/usr/bin/env python3
"""
Setup script for AI Third Umpire LBW Detection System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "outputs", "models"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("🚀 Setting up AI Third Umpire LBW Detection System")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed. Please check the error messages above.")
        return
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Get your Gemini AI API key from: https://makersuite.google.com/app/apikey")
    print("2. Run the application with: streamlit run app.py")
    print("3. Open your browser to the URL shown in the terminal")

if __name__ == "__main__":
    main()
