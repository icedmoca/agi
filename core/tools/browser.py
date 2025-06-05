def open_browser(url: str) -> dict:
    """Open a URL in the default browser with enhanced safety checks"""
    try:
        # Validate URL
        if not url or "{" in url or "[" in url or "result from step" in url.lower():
            return {
                "status": "skipped", 
                "output": f"⚠️ Skipped invalid or placeholder URL: {url}"
            }

        # Open URL safely
        import webbrowser
        webbrowser.open(url)
        
        return {
            "status": "success",
            "output": f"Opened {url}"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": str(e)
        } 