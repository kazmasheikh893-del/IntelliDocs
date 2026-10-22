"""
Tests for production access control, authentication gate, and setup_admin command.
"""

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import Document


class AccessControlTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="testadmin", password="TestPassword123!")

    def test_auth_disabled_by_default(self):
        """When REQUIRE_AUTH is False, views are publicly accessible."""
        with override_settings(REQUIRE_AUTH=False):
            r = self.client.get(reverse("documents:index"))
            self.assertEqual(r.status_code, 200)

            r_qa = self.client.get(reverse("documents:qa"))
            self.assertEqual(r_qa.status_code, 200)

    def test_anonymous_redirected_when_auth_required(self):
        """When REQUIRE_AUTH is True, anonymous users are redirected to login."""
        with override_settings(REQUIRE_AUTH=True):
            r = self.client.get(reverse("documents:index"))
            self.assertEqual(r.status_code, 302)
            self.assertIn(reverse("login"), r.url)

            r_qa = self.client.get(reverse("documents:qa"))
            self.assertEqual(r_qa.status_code, 302)
            self.assertIn(reverse("login"), r_qa.url)

            # JSON API request returns 401
            r_api = self.client.post(
                reverse("documents:api_query"),
                data=json.dumps({"question": "test"}),
                content_type="application/json",
            )
            self.assertEqual(r_api.status_code, 401)
            self.assertIn("Authentication required", r_api.json()["error"])

    def test_authenticated_user_has_access(self):
        """Logged in users can access protected pages when REQUIRE_AUTH is True."""
        with override_settings(REQUIRE_AUTH=True):
            self.client.login(username="testadmin", password="TestPassword123!")
            r = self.client.get(reverse("documents:index"))
            self.assertEqual(r.status_code, 200)

            r_qa = self.client.get(reverse("documents:qa"))
            self.assertEqual(r_qa.status_code, 200)

    def test_login_page_renders_cleanly(self):
        r = self.client.get(reverse("login"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Sign In to IntelliDocs")


class SetupAdminCommandTests(TestCase):
    def test_setup_admin_without_env_vars_creates_default_admin(self):
        with patch.dict("os.environ", {}, clear=True):
            call_command("setup_admin")
        User = get_user_model()
        u = User.objects.get(username="admin")
        self.assertTrue(u.is_superuser)
        self.assertTrue(u.check_password("Admin@12345"))

    def test_setup_admin_with_env_vars_creates_superuser(self):
        env = {
            "ADMIN_USERNAME": "superadmin",
            "ADMIN_PASSWORD": "SuperSecurePassword999!",
            "ADMIN_EMAIL": "super@intellidocs.ai",
        }
        with patch.dict("os.environ", env):
            call_command("setup_admin")

        User = get_user_model()
        u = User.objects.get(username="superadmin")
        self.assertTrue(u.is_superuser)
        self.assertTrue(u.is_staff)
        self.assertTrue(u.check_password("SuperSecurePassword999!"))

        # Re-run is idempotent
        with patch.dict("os.environ", env):
            call_command("setup_admin")
        self.assertEqual(User.objects.filter(username="superadmin").count(), 1)

