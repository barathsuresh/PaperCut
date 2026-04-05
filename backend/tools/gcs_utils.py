from google.auth.exceptions import RefreshError, TransportError as GoogleAuthTransportError


def is_gcs_transport_error(exc: Exception) -> bool:
    return isinstance(exc, (GoogleAuthTransportError, RefreshError))
