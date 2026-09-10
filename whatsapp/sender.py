"""
WhatsApp Sender — Fase 1 stub.
Sends WhatsApp messages via configured provider (Twilio or Meta Cloud API).
"""
# TODO Fase 1: implement Twilio client.


def send_message(to_phone: str, message: str, template: str | None = None) -> dict:
    """
    Sends a WhatsApp message.

    Args:
        to_phone:  Recipient phone in E.164 format, e.g. "+5491112345678"
        message:   Message body (for non-template messages)
        template:  Template name (required for proactive messages with Meta)

    Returns:
        dict with keys: provider_message_id, status
    """
    raise NotImplementedError("WhatsApp sender not yet implemented (Fase 1)")
