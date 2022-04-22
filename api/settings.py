from pydantic import BaseSettings, Field, RedisDsn


class Settings(BaseSettings):
    """

    """

    CELERY_BROKER_URL: RedisDsn = Field("amqp://", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: RedisDsn = Field("redis://", env="CELERY_RESULT_BACKEND")
