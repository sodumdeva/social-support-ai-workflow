"""
Tesseract OCR Installation Helper

This script helps install Tesseract OCR for document image processing
in the Social Support AI Workflow.
"""
import subprocess
import sys
import platform
import os


def check_tesseract_installed():
    """Check if Tesseract is already installed"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract is already installed: {version}")
        return True
    except ImportError:
        print("❌ pytesseract Python package not found")
        return False
    except Exception as e:
        print(f"❌ Tesseract not found: {str(e)}")
        return False


def install_tesseract():
    """Install Tesseract based on the operating system"""
    
    system = platform.system().lower()
    
    print(f"🔍 Detected OS: {system}")
    
    if system == "darwin":  # macOS
        print("📦 Installing Tesseract on macOS using Homebrew...")
        
        # Check if Homebrew is installed
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            print("✅ Homebrew found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
        
        # Install Tesseract
        try:
            print("   Installing tesseract...")
            result = subprocess.run(["brew", "install", "tesseract"], check=True, capture_output=True, text=True)
            print("✅ Tesseract installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Tesseract: {e}")
            print(f"   Error output: {e.stderr}")
            return False
    
    elif system == "linux":
        print("📦 Installing Tesseract on Linux...")
        
        # Try different package managers
        package_managers = [
            (["apt-get", "update"], ["apt-get", "install", "-y", "tesseract-ocr"]),
            (["yum", "update"], ["yum", "install", "-y", "tesseract"]),
            (["dnf", "update"], ["dnf", "install", "-y", "tesseract"]),
        ]
        
        for update_cmd, install_cmd in package_managers:
            try:
                print(f"   Trying {install_cmd[0]}...")
                subprocess.run(update_cmd, check=True, capture_output=True)
                subprocess.run(install_cmd, check=True, capture_output=True)
                print("✅ Tesseract installed successfully!")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print("❌ Could not install Tesseract automatically.")
        print("   Please install manually:")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   CentOS/RHEL: sudo yum install tesseract")
        print("   Fedora: sudo dnf install tesseract")
        return False
    
    elif system == "windows":
        print("📦 Windows detected")
        print("❌ Automatic installation not supported on Windows.")
        print("   Please install Tesseract manually:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Run the installer")
        print("   3. Add Tesseract to your PATH")
        print("   4. Restart your terminal/IDE")
        return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False


def install_python_packages():
    """Install required Python packages"""
    
    packages = ["pytesseract", "opencv-python", "pillow"]
    
    print("📦 Installing Python packages...")
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True


def test_installation():
    """Test if Tesseract is working correctly"""
    
    print("🧪 Testing Tesseract installation...")
    
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 10), "Test 123", fill='black', font=font)
        
        # Save test image
        test_image_path = "test_ocr.png"
        img.save(test_image_path)
        
        # Test OCR
        text = pytesseract.image_to_string(img)
        
        # Clean up
        os.remove(test_image_path)
        
        if "test" in text.lower() or "123" in text:
            print("✅ Tesseract is working correctly!")
            print(f"   Extracted text: '{text.strip()}'")
            return True
        else:
            print("⚠️  Tesseract installed but may not be working optimally")
            print(f"   Extracted text: '{text.strip()}'")
            return True
            
    except Exception as e:
        print(f"❌ Tesseract test failed: {str(e)}")
        return False


def main():
    """Main installation process"""
    
    print("🤖 Social Support AI - Tesseract OCR Installation")
    print("=" * 50)
    
    # Check if already installed
    if check_tesseract_installed():
        print("\n🎉 Tesseract is already working!")
        
        # Test it anyway
        if test_installation():
            print("\n✅ All set! You can now process document images.")
            return
    
    print("\n📦 Installing Tesseract OCR...")
    
    # Install Python packages first
    if not install_python_packages():
        print("\n❌ Failed to install Python packages")
        return
    
    # Install Tesseract
    if not install_tesseract():
        print("\n❌ Failed to install Tesseract")
        print("\n📋 Manual Installation Instructions:")
        print("macOS: brew install tesseract")
        print("Ubuntu: sudo apt-get install tesseract-ocr")
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return
    
    # Test installation
    print("\n🧪 Testing installation...")
    if test_installation():
        print("\n🎉 Installation successful!")
        print("✅ You can now upload and process document images in the Social Support AI application.")
    else:
        print("\n⚠️  Installation completed but testing failed.")
        print("   You may need to restart your terminal or check your PATH.")


if __name__ == "__main__":
    main() 