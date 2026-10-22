import os
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Creates or updates the production admin user from ADMIN_USERNAME and ADMIN_PASSWORD "
        "environment variables without ever printing the password to logs."
    )

    def handle(self, *args, **options):
        username = (os.environ.get("ADMIN_USERNAME") or "").strip()
        password = (os.environ.get("ADMIN_PASSWORD") or "").strip()
        email = (os.environ.get("ADMIN_EMAIL") or "admin@example.com").strip()

        if not username or not password:
            self.stdout.write(
                "ADMIN_USERNAME or ADMIN_PASSWORD not configured. Skipping automated admin creation."
            )
            return

        User = get_user_model()
        user, created = User.objects.get_or_create(
            username=username,
            defaults={"email": email, "is_staff": True, "is_superuser": True},
        )
        user.set_password(password)
        user.is_staff = True
        user.is_superuser = True
        user.save()

        if created:
            self.stdout.write(self.style.SUCCESS(f"Successfully created administrative user '{username}'."))
        else:
            self.stdout.write(self.style.SUCCESS(f"Successfully updated credentials for user '{username}'."))

