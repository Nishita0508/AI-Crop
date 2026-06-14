"""
routes/dashboard_routes.py
Handles the user dashboard (overview of recent activity) and
the full history page (all past predictions).
"""

from flask import Blueprint, render_template, session
from bson.objectid import ObjectId

from db import users_collection, crop_predictions_collection, disease_predictions_collection
from utils.auth_utils import login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    user = users_collection.find_one({"_id": ObjectId(user_id)})

    # ---- Recent crop recommendations (latest 5) ----
    recent_crops = list(
        crop_predictions_collection.find({"user_id": user_id})
        .sort("created_at", -1)
        .limit(5)
    )

    # ---- Recent disease predictions (latest 5) ----
    recent_diseases = list(
        disease_predictions_collection.find({"user_id": user_id})
        .sort("created_at", -1)
        .limit(5)
    )

    # ---- Total prediction count ----
    total_crop_count = crop_predictions_collection.count_documents({"user_id": user_id})
    total_disease_count = disease_predictions_collection.count_documents({"user_id": user_id})
    total_predictions = total_crop_count + total_disease_count

    return render_template(
        "dashboard.html",
        user=user,
        recent_crops=recent_crops,
        recent_diseases=recent_diseases,
        total_predictions=total_predictions,
        total_crop_count=total_crop_count,
        total_disease_count=total_disease_count,
    )


@dashboard_bp.route("/history")
@login_required
def history():
    user_id = session["user_id"]

    all_crops = list(
        crop_predictions_collection.find({"user_id": user_id}).sort("created_at", -1)
    )
    all_diseases = list(
        disease_predictions_collection.find({"user_id": user_id}).sort("created_at", -1)
    )

    return render_template(
        "history.html",
        all_crops=all_crops,
        all_diseases=all_diseases,
    )
