from __future__ import annotations

import os
from flask import Flask, jsonify, request

from logic import choose_move, compare_models, get_info

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(get_info())


@app.post("/start")
def start():
    return jsonify({})


@app.post("/move")
def move():
    game_state = request.get_json(force=True, silent=True) or {}
    return jsonify({"move": choose_move(game_state)})


@app.post("/end")
def end():
    return jsonify({})


@app.post("/debug/compare")
def debug_compare():
    game_state = request.get_json(force=True, silent=True) or {}
    return jsonify(compare_models(game_state))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
