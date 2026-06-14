"""
utils/auth_utils.py
Authentication helper functions: password hashing/verification
and a login_required decorator for protecting routes.
"""

from functools import wraps
from flask import session, redirect, url_for, flash
import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt. Returns a UTF-8 string."""
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def login_required(view_func):
    """Decorator to protect routes that require an authenticated user."""
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))
        return view_func(*args, **kwargs)
    return wrapped_view
