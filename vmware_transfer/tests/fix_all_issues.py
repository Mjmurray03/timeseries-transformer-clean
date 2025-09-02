import json
from pathlib import Path
import re

def fix_all_issues():
    """Fix all system issues at once"""
    
    print("=" * 60)
    print("FIXING ALL SYSTEM ISSUES")
    print("=" * 60)
    
    # Fix 1: Update API validation
    print("\n[1] Fixing API validation...")
    api_file = Path("src/api/main.py")
    
    if api_file.exists():
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Fix the PredictionRequest model if it exists
        content = re.sub(
            r'must have exactly 8 features',
            'must have exactly 10 features',
            content
        )
        
        # Fix any hardcoded 8s related to features
        content = re.sub(
            r'len\(features\[0\]\) != 8',
            'len(features[0]) != 10',
            content
        )
        
        # Look for the validation in PredictionRequest class
        if 'class PredictionRequest' in content:
            # Find and update the validation
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'features[0]' in line and '8' in line:
                    lines[i] = line.replace('8', '10')
            content = '\n'.join(lines)
        
        with open(api_file, 'w') as f:
            f.write(content)
        
        print("  Fixed API validation to expect 10 features")
    else:
        print("  API main.py not found")
    
    # Fix 2: Check if API schemas need updating
    print("\n[2] Checking API schemas...")
    schemas_file = Path("src/api/schemas.py")
    if schemas_file.exists():
        with open(schemas_file, 'r') as f:
            schemas_content = f.read()
        
        schemas_content = re.sub(r'exactly 8 features', 'exactly 10 features', schemas_content)
        schemas_content = re.sub(r'!= 8', '!= 10', schemas_content)
        
        with open(schemas_file, 'w') as f:
            f.write(schemas_content)
        
        print("  Fixed API schemas")
    else:
        print("  No schemas.py file found")
    
    # Fix 3: Verify scalers are correct
    print("\n[3] Verifying scalers...")
    
    for ticker in ['AAPL', 'MSFT', 'GOOG']:
        scaler_path = Path(f"scalers/scaler_{ticker}.json")
        if scaler_path.exists():
            with open(scaler_path) as f:
                scaler = json.load(f)
            
            if len(scaler.get('feat_mean', [])) == 10:
                print(f"  {ticker} scaler: OK (10 features)")
            else:
                print(f"  {ticker} scaler: NEEDS FIX")
        else:
            print(f"  {ticker} scaler: NOT FOUND")
    
    # Fix 4: Check for other validation files
    print("\n[4] Checking for other validation files...")
    
    # Check if there are other files that might have feature validation
    for file_path in Path("src").rglob("*.py"):
        if file_path.name in ['__pycache__']:
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            if '8 features' in content or 'features[0]) != 8' in content:
                print(f"  Found potential issue in: {file_path}")
                
                # Apply the same fixes
                content = re.sub(r'8 features', '10 features', content)
                content = re.sub(r'!= 8', '!= 10', content)
                
                with open(file_path, 'w') as f:
                    f.write(content)
                    
                print(f"  Fixed: {file_path}")
        except:
            # Skip files that can't be read
            continue
    
    print("\n" + "=" * 60)
    print("FIX SUMMARY")
    print("=" * 60)
    print("\n[SUCCESS] All automatic fixes applied")
    print("\nNext steps:")
    print("1. Restart the API server if it's running")
    print("2. Run: python tests/comprehensive_diagnostic.py")
    print("3. Run: python tests/test_realistic_prediction.py")
    print("4. If issues persist, check the API logs for detailed errors")
    
    return True

if __name__ == "__main__":
    fix_all_issues()