import logging

logger = logging.getLogger(__name__)

def get_active_config(config_module=None):
    """
    Returns the appropriate config module (passed-in or default).

    Args:
        config_module: An optional config module (e.g., test_config).

    Returns:
        The active config module to use.
    """
    if config_module:
        logger.info(f"Using provided config module: {config_module.__name__}")
        return config_module
    else:
        # If no module is passed, import and return the default config package
        import config # Import the default config package
        logger.info("Using default config package.")
        return config