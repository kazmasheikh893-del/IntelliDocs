"""
Django settings for intellidocs project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
load_dotenv(BASE_DIR / '.env')

# SECURITY WARNING: keep the secret key used in production secret!
# DEBUG defaults to False (safe); set DEBUG=True in your local .env for development.
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 'yes')

SECRET_KEY = os.environ.get('SECRET_KEY', '')
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = 'django-insecure-local-development-only'
    else:
        # Fallback to a cryptographically secure key if not explicitly set, preventing startup crash
        import secrets
        SECRET_KEY = secrets.token_urlsafe(50)

ALLOWED_HOSTS = [h.strip() for h in os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1,.onrender.com,.railway.app').split(',') if h.strip()]

# Auto-detect Render external hostname if running on Render
RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
if RENDER_EXTERNAL_HOSTNAME and RENDER_EXTERNAL_HOSTNAME not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

# Access control and authentication gate
# When True, protects upload, query, and delete operations from unauthorized public internet users.
REQUIRE_AUTH = os.environ.get('REQUIRE_AUTH', 'False' if DEBUG else 'True').lower() in ('true', '1', 'yes')
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'documents:index'
LOGOUT_REDIRECT_URL = 'login'

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'documents.apps.DocumentsConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'intellidocs.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'intellidocs.wsgi.application'

# Database and persistent storage configuration
# If DATA_DIR is provided (e.g. persistent volume mount), store db.sqlite3 and uploads there.
DATA_DIR_PATH = os.environ.get('DATA_DIR', '').strip()
DATA_DIR = Path(DATA_DIR_PATH) if DATA_DIR_PATH else BASE_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': DATA_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL = '/media/'
MEDIA_ROOT = DATA_DIR / 'media'
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


STATIC_ROOT = BASE_DIR / 'staticfiles'


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Google Gemini API configuration (official google-genai SDK) ---------------
# The key is read ONLY from the environment / .env and is never sent to templates.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
# text-embedding-004 was retired by Google; gemini-embedding-001 is its supported successor.
GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
GEMINI_EMBEDDING_DIM = _env_int('GEMINI_EMBEDDING_DIM', 768)
GEMINI_GENERATION_MODEL = os.environ.get('GEMINI_GENERATION_MODEL', 'gemini-3.5-flash-lite')

# --- Retrieval / chunking parameters --------------------------------------------
RAG_TOP_K = _env_int('RAG_TOP_K', 5)
RAG_MIN_SIMILARITY = _env_float('RAG_MIN_SIMILARITY', 0.45)
RAG_CHUNK_SIZE = _env_int('RAG_CHUNK_SIZE', 800)
RAG_CHUNK_OVERLAP = _env_int('RAG_CHUNK_OVERLAP', 100)
RAG_MAX_QUESTION_CHARS = 1000

# --- Upload limits --------------------------------------------------------------
MAX_UPLOAD_SIZE_BYTES = _env_int('MAX_UPLOAD_SIZE_MB', 10) * 1024 * 1024
ALLOWED_FILE_EXTENSIONS = ['.pdf', '.txt']

# --- Logging (never log keys or document text) ----------------------------------
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {'console': {'class': 'logging.StreamHandler'}},
    'loggers': {'documents': {'handlers': ['console'], 'level': 'INFO'}},
}

# --- Production hardening (applies when DEBUG is off) ---------------------------
if not DEBUG:
    SECURE_SSL_REDIRECT = os.environ.get('SECURE_SSL_REDIRECT', 'True').lower() in ('true', '1', 'yes')
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = _env_int('SECURE_HSTS_SECONDS', 3600)
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    _csrf_origins = os.environ.get('CSRF_TRUSTED_ORIGINS', 'https://*.onrender.com,https://*.railway.app')
    CSRF_TRUSTED_ORIGINS = [o.strip() for o in _csrf_origins.split(',') if o.strip()]
    if RENDER_EXTERNAL_HOSTNAME:
        render_origin = f"https://{RENDER_EXTERNAL_HOSTNAME}"
        if render_origin not in CSRF_TRUSTED_ORIGINS:
            CSRF_TRUSTED_ORIGINS.append(render_origin)

# Serve collected static files in production (run collectstatic on deploy).
if not DEBUG:
    MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
    STORAGES = {
        'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
        'staticfiles': {'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage'},
    }
