
from flask import Blueprint, jsonify, request, Response
brnn_predict_blueprint = Blueprint(f"brnn_predict", __name__)


@brnn_predict_blueprint.get("/BRNN/predict")
def get_predict() -> Response:
    """One mandatory 'source code' parameter must be passed
        http://127.0.0.1:5000/BRNN/predict?source_code=code
    """
    import model
    source_code = request.args.get('source_code')
    predictions = model.predict(source_code)
    return jsonify(predictions)

