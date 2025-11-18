"""
Very lightweight monitoring stub for logging requests.
In a real system, this would push to Prometheus / Grafana.
"""

import datetime

REQUEST_LOG = []


def log_request(endpoint: str):
    REQUEST_LOG.append(
        {
            "endpoint": endpoint,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    )
