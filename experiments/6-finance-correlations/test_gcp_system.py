#!/usr/bin/env python3
"""
Test script for the GCP Real-Time Market Prediction System
Tests data collection, Max[Z] calculation, and prediction functionality.
"""

import sys
import os
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GCP system components
from gcp_finance_analysis import (
    Config, DataBuffer, GCPDataCollector, MarketDataCollector, 
    PredictionEngine, PredictionTracker, MaxZCalculator
)

def test_gcp_data_collection():
    """Test GCP data collection from the API."""
    print("🧪 Testing GCP Data Collection...")
    
    # Initialize buffer and collector
    buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
    collector = GCPDataCollector(buffer)
    
    try:
        # Test single data fetch
        print("  📡 Fetching GCP data from global-mind.org...")
        collector._fetch_gcp_data()
        
        # Check if we got data
        data_count = buffer.size()
        print(f"  ✅ Collected {data_count} GCP data points")
        
        if data_count > 0:
            # Show sample data
            recent_data = buffer.get_recent(5)
            print("  📊 Sample data:")
            for i, data_point in enumerate(recent_data):
                print(f"    {i+1}. {data_point['egg_id']}: {data_point['value']} @ {data_point['timestamp']}")
        
        return data_count > 0
        
    except Exception as e:
        print(f"  ❌ GCP data collection failed: {e}")
        return False

def test_max_z_calculation():
    """Test Max[Z] calculation with sample data."""
    print("\n🧪 Testing Max[Z] Calculation...")
    
    # Create sample GCP data
    sample_data = [
        {'timestamp': datetime.now(), 'egg_id': 'egg_1', 'value': 105, 'site_id': 'egg_1'},
        {'timestamp': datetime.now(), 'egg_id': 'egg_2', 'value': 95, 'site_id': 'egg_2'},
        {'timestamp': datetime.now(), 'egg_id': 'egg_3', 'value': 110, 'site_id': 'egg_3'},
        {'timestamp': datetime.now(), 'egg_id': 'egg_4', 'value': 90, 'site_id': 'egg_4'},
        {'timestamp': datetime.now(), 'egg_id': 'egg_5', 'value': 115, 'site_id': 'egg_5'},
    ]
    
    try:
        max_z = MaxZCalculator.calculate_max_z(sample_data)
        print(f"  ✅ Max[Z] calculated: {max_z:.4f}")
        
        # Test with expected GCP values
        expected_max_z = (115 - 100) / 7.0712  # Should be ~2.12
        print(f"  📊 Expected Max[Z] for value 115: {expected_max_z:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Max[Z] calculation failed: {e}")
        return False

def test_market_data_collection():
    """Test market data collection via Alpaca API."""
    print("\n🧪 Testing Market Data Collection...")
    
    # Load .env file and check if Alpaca credentials are available
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("  ⚠️  Alpaca credentials not found in .env file")
        print("  📝 Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env to test market data")
        return False
    
    try:
        # Initialize buffer and collector
        buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
        collector = MarketDataCollector(buffer)
        
        # Test single data fetch
        print("  📡 Fetching market data from Alpaca...")
        
        # Fetch all symbols together (like get_prices.py)
        for symbol in Config.MARKET_SYMBOLS:
            try:
                trade = collector.api.get_latest_trade(symbol)
                data_point = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'price': trade.price,
                    'volume': trade.size
                }
                buffer.add(data_point)
                print(f"    ✅ {symbol}: ${trade.price:.2f}")
            except Exception as e:
                print(f"    ❌ {symbol}: {e}")
        
        # Check if we got data
        data_count = buffer.size()
        print(f"  ✅ Collected {data_count} market data points")
        
        if data_count > 0:
            # Show sample data
            recent_data = buffer.get_recent(3)
            print("  📊 Sample market data:")
            for i, data_point in enumerate(recent_data):
                print(f"    {i+1}. {data_point['symbol']}: ${data_point['price']:.2f} @ {data_point['timestamp']}")
        
        return data_count > 0
        
    except Exception as e:
        print(f"  ❌ Market data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_engine():
    """Test the prediction engine with sample data."""
    print("\n🧪 Testing Prediction Engine...")
    
    try:
        # Initialize components
        gcp_buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
        market_buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
        tracker = PredictionTracker()
        engine = PredictionEngine(gcp_buffer, market_buffer, tracker)
        
        # Add some sample GCP data
        sample_gcp_data = [
            {'timestamp': datetime.now(), 'egg_id': 'egg_1', 'value': 120, 'site_id': 'egg_1'},
            {'timestamp': datetime.now(), 'egg_id': 'egg_2', 'value': 85, 'site_id': 'egg_2'},
            {'timestamp': datetime.now(), 'egg_id': 'egg_3', 'value': 110, 'site_id': 'egg_3'},
        ]
        
        for data_point in sample_gcp_data:
            gcp_buffer.add(data_point)
        
        # Test prediction generation
        print("  🎯 Testing prediction generation...")
        prediction, confidence = engine._predict_direction(2.5, 'SPY')
        print(f"  ✅ Prediction: {prediction} (confidence: {confidence:.2f})")
        
        # Test prediction tracking
        print("  📊 Testing prediction tracking...")
        tracker.add_prediction('SPY', prediction, confidence)
        stats = tracker.get_stats('SPY', hours=1)
        print(f"  ✅ Tracker stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction engine test failed: {e}")
        return False

def test_full_system():
    """Test the complete system integration."""
    print("\n🧪 Testing Full System Integration...")
    
    try:
        # Initialize all components
        gcp_buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
        market_buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
        tracker = PredictionTracker()
        
        gcp_collector = GCPDataCollector(gcp_buffer)
        market_collector = MarketDataCollector(market_buffer)
        prediction_engine = PredictionEngine(gcp_buffer, market_buffer, tracker)
        
        print("  🚀 Starting system components...")
        
        # Start collectors
        gcp_collector.start()
        market_collector.start()
        prediction_engine.start()
        
        # Let it run for enough time to make predictions
        print("  ⏱️  Running system for 35 seconds...")
        time.sleep(35)
        
        # Stop everything
        print("  🛑 Stopping system components...")
        gcp_collector.stop()
        market_collector.stop()
        prediction_engine.stop()
        
        # Check results
        gcp_count = gcp_buffer.size()
        market_count = market_buffer.size()
        prediction_count = len(tracker.predictions)
        
        print(f"  📊 System Results:")
        print(f"    - GCP data points: {gcp_count}")
        print(f"    - Market data points: {market_count}")
        print(f"    - Predictions made: {prediction_count}")
        
        # Debug: Check recent GCP data timing
        if gcp_count > 0:
            recent_gcp = gcp_buffer.get_recent(5)
            print(f"    - Recent GCP timestamps: {[dp['timestamp'] for dp in recent_gcp]}")
        
        # Debug: Check if prediction engine has enough data
        if gcp_count > 0:
            time_window_data = gcp_buffer.get_time_window(Config.BIN_DURATION_SECONDS)
            print(f"    - GCP data in {Config.BIN_DURATION_SECONDS}s window: {len(time_window_data)}")
            print(f"    - Min samples needed: {Config.MIN_SAMPLES_FOR_PREDICTION}")
        
        if gcp_count > 0 and prediction_count > 0:
            print("  ✅ Full system test PASSED")
            return True
        else:
            print("  ⚠️  Full system test - limited data collected")
            return False
        
    except Exception as e:
        print(f"  ❌ Full system test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 GCP Real-Time Market Prediction System - Test Suite")
    print("=" * 60)
    
    # Run individual tests
    tests = [
        ("GCP Data Collection", test_gcp_data_collection),
        ("Max[Z] Calculation", test_max_z_calculation),
        ("Market Data Collection", test_market_data_collection),
        ("Prediction Engine", test_prediction_engine),
        ("Full System Integration", test_full_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to run.")
    elif passed >= 3:
        print("⚠️  Most tests passed. System should work with some limitations.")
    else:
        print("❌ Multiple tests failed. Please check configuration and dependencies.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 