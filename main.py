
import sys
from src.exercises import BicepCurlChecker
from src.utils.logger import AppLogger
from src.utils.validators import InputValidator

logger = AppLogger.get_logger(__name__)


def main():
    """Main entry point for the application."""
    logger.info("Application started")
    
    print("\n" + "=" * 60)
    print("AI GYM FORM CHECKER")
    print("=" * 60)
    print("\nAvailable Exercises:")
    print("  1. Bicep Curl")
    print("  (More exercises coming soon!)")
    print("=" * 60)
    
    # Get validated user input
    choice = InputValidator.get_exercise_choice()
    
    if choice is None:
        logger.info("User cancelled exercise selection")
        print("\nExiting... Goodbye!")
        return
    
    if choice == 'bicep_curl':
        logger.info("Starting Bicep Curl exercise")
        try:
            app = BicepCurlChecker()
            app.run()
        except Exception as e:
            logger.error(f"Error running Bicep Curl exercise: {e}", exc_info=True)
            print(f"\nError: {e}")
            print("Please check logs/gym_*.log for details.")
    else:
        logger.warning(f"Unknown exercise choice: {choice}")
        print(f"Exercise '{choice}' not yet implemented.")


if __name__ == "__main__":
    try:
        main()
        logger.info("Application exited normally")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\nExiting... Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        print(f"\n\nCritical Error: {e}")
        print("Please check:")
        print("  1. Camera is connected and working")
        print("  2. All dependencies are installed (pip install -r requirements.txt)")
        print("  3. Check logs/gym_*.log for detailed error information")
        sys.exit(1)
