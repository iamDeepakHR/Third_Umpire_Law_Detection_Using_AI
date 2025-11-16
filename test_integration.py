#!/usr/bin/env python3
"""
Test script to verify the enhanced LBW detection system integration
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
        
        import google.generativeai as genai
        print("✓ Google Generative AI imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
        
        from lbw_ai.explainer import generate_explanation, ExplanationInputs
        print("✓ LBW AI explainer imported successfully")
        
        from lbw_ai.visualize import create_complete_trajectory_image
        print("✓ LBW AI visualization imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_gemini_integration():
    """Test Gemini AI integration (without API key)"""
    try:
        from lbw_ai.explainer import generate_explanation, ExplanationInputs
        
        # Test basic explanation without API key
        inputs = ExplanationInputs(
            pitched_zone="in-line",
            impact_in_line=True,
            would_hit_stumps=True,
            decision="OUT",
            model_confidence=0.85,
            track_points=[(100, 200), (150, 250), (200, 300)],
            future_points=[(250, 350), (300, 400)],
            bounce_index=1,
            distance_to_stumps_px=50.0
        )
        
        explanation = generate_explanation(inputs, use_ai=False)
        print("✓ Basic explanation generation works")
        print(f"  Sample explanation: {explanation[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Gemini integration test failed: {e}")
        return False

def test_visualization():
    """Test trajectory visualization"""
    try:
        from lbw_ai.visualize import create_complete_trajectory_image
        
        # Create sample data
        track_points = [(100, 200), (150, 250), (200, 300), (250, 350)]
        predicted_points = [(300, 400), (350, 450), (400, 500)]
        stumps_x_left = 380
        stumps_x_right = 420
        
        # Generate trajectory image
        trajectory_img = create_complete_trajectory_image(
            track=track_points,
            predicted=predicted_points,
            stumps_x_left=stumps_x_left,
            stumps_x_right=stumps_x_right,
            decision="OUT",
            bounce_index=2,
            frame_height=600,
            frame_width=800
        )
        
        print("✓ Trajectory visualization works")
        print(f"  Generated image shape: {trajectory_img.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Enhanced LBW Detection System Integration")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Gemini Integration Test", test_gemini_integration),
        ("Visualization Test", test_visualization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} FAILED")
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The integration is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
