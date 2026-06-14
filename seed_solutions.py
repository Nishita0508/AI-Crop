from db import disease_solutions_collection

solutions = [
    {"disease_name": "Apple - Apple Scab",
     "cause": "Fungal disease caused by Venturia inaequalis, favored by cool, wet spring weather.",
     "solution": "Apply fungicide sprays from bud break, rake and destroy fallen leaves, prune for airflow, and choose scab-resistant varieties."},

    {"disease_name": "Apple - Black Rot",
     "cause": "Caused by the fungus Botryosphaeria obtusa, which enters through wounds and dead wood.",
     "solution": "Prune out cankers and dead wood, remove mummified fruit, and apply fungicide during the growing season."},

    {"disease_name": "Apple - Cedar Apple Rust",
     "cause": "Fungal disease that alternates between apple and nearby juniper/cedar trees.",
     "solution": "Remove nearby junipers/cedars if possible, apply preventive fungicide in spring, and plant rust-resistant varieties."},

    {"disease_name": "Apple - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue regular watering, fertilization, and monitoring."},

    {"disease_name": "Bell Pepper - Bacterial Spot",
     "cause": "Caused by Xanthomonas bacteria, spread by water splash, rain, and contaminated tools or seed.",
     "solution": "Use disease-free seed, avoid overhead irrigation, apply copper-based bactericides, and rotate crops for 2-3 years."},

    {"disease_name": "Bell Pepper - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Maintain consistent watering and balanced fertilization."},

    {"disease_name": "Cherry - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue routine pruning and pest monitoring."},

    {"disease_name": "Cherry - Powdery Mildew",
     "cause": "Fungal disease favored by warm days, cool nights, and high humidity.",
     "solution": "Improve air circulation through pruning, apply sulfur or approved fungicides, and avoid excess nitrogen fertilizer."},

    {"disease_name": "Corn (Maize) - Cercospora Leaf Spot",
     "cause": "Fungal disease (gray leaf spot) that thrives in warm, humid conditions with extended leaf wetness.",
     "solution": "Rotate crops, plant resistant hybrids, manage crop residue, and apply foliar fungicide if disease pressure is high."},

    {"disease_name": "Corn (Maize) - Common Rust",
     "cause": "Fungal disease caused by Puccinia sorghi, spread by windborne spores.",
     "solution": "Plant resistant hybrids, monitor fields regularly, and apply fungicide if infection is early and severe."},

    {"disease_name": "Corn (Maize) - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Maintain proper fertilization and irrigation."},

    {"disease_name": "Corn (Maize) - Northern Leaf Blight",
     "cause": "Fungal disease favored by moderate temperatures and high humidity.",
     "solution": "Use resistant hybrids, rotate crops, till under crop residue, and apply fungicide if needed."},

    {"disease_name": "Grape - Black Rot",
     "cause": "Fungal disease that spreads rapidly in warm, wet weather.",
     "solution": "Remove mummified berries and infected canes, apply fungicide from early shoot growth through fruit set, and improve canopy airflow."},

    {"disease_name": "Grape - Esca (Black Measles)",
     "cause": "Caused by a complex of wood-rotting fungi entering through pruning wounds.",
     "solution": "Avoid pruning in wet weather, protect pruning cuts, and remove severely infected vines. No curative chemical treatment exists."},

    {"disease_name": "Grape - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue regular canopy management and monitoring."},

    {"disease_name": "Grape - Leaf Blight",
     "cause": "Fungal leaf spot disease favored by warm, humid conditions.",
     "solution": "Improve canopy ventilation, remove infected leaves, and apply fungicide preventively."},

    {"disease_name": "Peach - Bacterial Spot",
     "cause": "Caused by Xanthomonas bacteria, spread by rain splash and wind.",
     "solution": "Use resistant varieties, apply copper-based sprays during dormancy, avoid overhead irrigation, and prune for airflow."},

    {"disease_name": "Peach - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue regular care, watering, and pest monitoring."},

    {"disease_name": "Potato - Early Blight",
     "cause": "Fungal disease caused by Alternaria solani, favored by warm temperatures and high humidity.",
     "solution": "Rotate crops, remove infected debris, apply fungicide, avoid overhead watering, and maintain plant nutrition."},

    {"disease_name": "Potato - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue regular irrigation and monitoring."},

    {"disease_name": "Potato - Late Blight",
     "cause": "Caused by Phytophthora infestans, which spreads rapidly in cool, wet conditions.",
     "solution": "Plant resistant varieties, apply fungicide preventively, destroy infected plants promptly, and avoid overhead irrigation."},

    {"disease_name": "Strawberry - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue regular watering and bed renovation practices."},

    {"disease_name": "Strawberry - Leaf Scorch",
     "cause": "Fungal disease that spreads in warm, wet conditions.",
     "solution": "Remove infected leaves after harvest, improve air circulation, apply fungicide if needed, and avoid overhead irrigation."},

    {"disease_name": "Tomato - Bacterial Spot",
     "cause": "Caused by Xanthomonas species, spread through contaminated seed, water splash, and tools.",
     "solution": "Use certified disease-free seed/transplants, apply copper-based bactericides, avoid handling wet plants, and rotate crops."},

    {"disease_name": "Tomato - Early Blight",
     "cause": "Fungal disease caused by Alternaria solani, common in warm humid weather, often starting on lower leaves.",
     "solution": "Remove infected leaves, mulch to reduce soil splash, apply fungicide, rotate crops, and stake plants for airflow."},

    {"disease_name": "Tomato - Healthy",
     "cause": "No disease detected.",
     "solution": "Plant appears healthy. Continue balanced watering and fertilization."},

    {"disease_name": "Tomato - Late Blight",
     "cause": "Caused by Phytophthora infestans, which spreads quickly in cool, moist conditions and can destroy a crop within days.",
     "solution": "Apply fungicide preventively, remove and destroy infected plants immediately, avoid overhead watering, and ensure good ventilation."},

    {"disease_name": "Tomato - Septoria Leaf Spot",
     "cause": "Fungal disease that spreads via splashing water and favors humid conditions.",
     "solution": "Remove lower infected leaves, mulch around plants, apply fungicide, avoid overhead irrigation, and rotate crops."},

    {"disease_name": "Tomato - Yellow Leaf Curl Virus",
     "cause": "Viral disease transmitted by whiteflies, causing leaf curling, yellowing, and stunted growth.",
     "solution": "Control whiteflies with insecticides or sticky traps, use resistant varieties, remove infected plants, and use reflective mulches."},
]

for item in solutions:
    disease_solutions_collection.update_one(
        {"disease_name": item["disease_name"]},
        {"$set": item},
        upsert=True
    )

print(f"Inserted/updated {len(solutions)} disease solution records.")