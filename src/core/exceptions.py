"""Excepciones de dominio. La infraestructura las traduce; los controladores las mapean a HTTP."""


class DomainError(Exception):
    """Error de negocio genérico."""


class PatientNotFoundError(DomainError):
    """El identificador de paciente no existe en el repositorio."""


class InvalidPatientDataError(DomainError):
    """Los datos del paciente no cumplen el contrato mínimo para predecir."""


class DataAccessError(DomainError):
    """Fallo al leer o consolidar las fuentes de datos."""


class PredictionError(DomainError):
    """El modelo predictivo no pudo generar una probabilidad válida."""


class AnalyticsError(DomainError):
    """El analista conversacional no pudo completar la consulta."""
