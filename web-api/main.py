import sys
import logging

from flask_cors import CORS
from services.prediction import brnn_predict_blueprint
from flask import Flask, request, jsonify, Blueprint
from waitress import serve


def config_logger() -> logging.Logger:
    from applogger import AppLogger
    sys.stderr = sys.stdout  # the info messages are no longer showed on red from PyCharm
    AppLogger.config(logging.DEBUG)
    return AppLogger.getLogger()


def register_imported_blueprints(app: Flask) -> int:
    from applogger import AppLogger
    count = 0
    for variable_name, variable in globals().items():
        if variable_name.endswith("_blueprint") and isinstance(variable, Blueprint):
            app.register_blueprint(variable)
            count += 1
    AppLogger.getLogger().debug(f"Registered {count} blueprints in flask.")
    return count


def import_model() -> None:
    """Imports the model component from a sibling directory"""
    import os
    import sys
    if not os.path.exists("../common/model.py"):
        print("You need to include the 'common' directory as a sibling directory of 'web-api'.", file=sys.stderr)
        sys.exit(-1)
    sys.path.append('../common')
    import model  # loads the model into memory


def main() -> None:
    import_model()
    app = Flask(__name__)
    CORS(app)
    register_imported_blueprints(app)
    serve(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    config_logger()
    main()
