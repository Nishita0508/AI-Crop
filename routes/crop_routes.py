"""
routes/crop_routes.py
Handles crop recommendation: accepts soil/weather parameters,
predicts the best crop using the Scikit-learn model, and saves
the recommendation history to MongoDB.
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app

from db import crop_predictions_collection, get_server_time
from utils.auth_utils import login_required
from utils.validators import validate_crop_input
from utils.model_loader import predict_crop

crop_bp = Blueprint("crop", __name__)


@crop_bp.route("/predict-crop", methods=["GET", "POST"])
@login_required
def predict_crop_route():
    if request.method == "POST":
        form_data = request.form.to_dict()

        # ---- Validate input ----
        is_valid, errors, parsed = validate_crop_input(form_data)
        if not is_valid:
            for err in errors:
                flash(err, "danger")
            return render_template("crop.html", result=None)

        try:
            # ---- Run prediction ----
            crop_name = predict_crop(parsed)

            # ---- Save recommendation history ----
            recommendation_doc = {
                "user_id": session["user_id"],
                "inputs": parsed,
                "recommended_crop": crop_name,
                "created_at": get_server_time(),
            }
            crop_predictions_collection.insert_one(recommendation_doc)

            return render_template(
                "crop.html",
                result={
                    "recommended_crop": crop_name,
                    "inputs": parsed,
                }
            )

        except Exception as e:
            current_app.logger.error(f"Crop prediction error: {e}")
            flash("An error occurred while predicting the crop. Please try again.", "danger")
            return render_template("crop.html", result=None)

    return render_template("crop.html", result=None)
