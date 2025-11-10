#!/usr/bin/env python3
"""
Oral Lesion Detection Web Application Launcher
"""

from app import app
import os

if __name__ == '__main__':
    # Check for required model files
    required_files = ['best_hybrid_model.h5', 'tokenizer.pickle', 'class_names.pickle']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("❌ Missing required model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📁 Please ensure all model files are in the current directory.")
        exit(1)

    print("🚀 Starting Oral Lesion Detection Web Application...")
    print("🌐 Application will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")

    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)