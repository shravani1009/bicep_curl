
import sys
from src.exercises import BicepCurlChecker


def main():
    """Main entry point for the application."""
    print("\n" + "=" * 60)
    print("AI GYM FORM CHECKER")
    print("=" * 60)
    print("\nAvailable Exercises:")
    print("  1. Bicep Curl")
    print("  (More exercises coming soon!)")
    print("=" * 60)
    
    # For now, only bicep curl is available
    choice = input("\nSelect exercise (1 for Bicep Curl): ").strip()
    
    if choice == '1' or choice.lower() == 'bicep curl':
        app = BicepCurlChecker()
        app.run()
    else:
        print("Invalid choice! Defaulting to Bicep Curl...")
        app = BicepCurlChecker()
        app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        print("Please check your camera and dependencies.")
        sys.exit(1)
