import sys
import logging
from services.prediction import brnn_predict_blueprint

from flask import Flask, request, jsonify, Blueprint


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


def main() -> None:
    app = Flask(__name__)
    register_imported_blueprints(app)
    app.run(debug=True)  # debug must be False to trace programs in PyCharm


if __name__ == '__main__':
    config_logger()
    main()
