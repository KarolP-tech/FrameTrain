"""
Verify key command for FrameTrain CLI
"""

import sys
from ..utils import (
    print_success,
    print_error,
    print_info,
    make_api_request,
    format_date
)


def verify_api_key(api_key: str):
    """Verify an API key with the FrameTrain server"""
    
    print_info("Verifying API key...")
    
    try:
        response = make_api_request(
            '/keys/verify',
            method='POST',
            data={'key': api_key}
        )
        
        if response and response.get('valid'):
            print_success("\n✓ API Key is VALID")
            
            # Show additional info if available
            user_id = response.get('userId')
            expires_at = response.get('expiresAt')
            last_used = response.get('lastUsedAt')
            
            if user_id:
                print_info(f"User ID: {user_id}")
            
            if expires_at:
                print_info(f"Expires: {format_date(expires_at)}")
            else:
                print_info("Expires: Never")
            
            if last_used:
                print_info(f"Last used: {format_date(last_used)}")
            
        else:
            print_error("\n✗ API Key is INVALID")
            message = response.get('message', 'Unknown error')
            print_info(f"Reason: {message}")
            sys.exit(1)
            
    except Exception as e:
        print_error(f"\n✗ Verification failed: {str(e)}")
        print_info("Make sure you have an internet connection and the API server is reachable")
        sys.exit(1)
