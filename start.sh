#!/usr/bin/env bash
exec gunicorn clubTest:app --bind 0.0.0.0:${PORT:-5000} --workers 2
