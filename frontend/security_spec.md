# Security Specification — FisuraCare Clinical Analytics

## 1. Data Invariants
1. **Strict Ownership & PII Isolation**: A specialist profile at `/specialists/{specialistId}` can only be read or written by the authenticated, email-verified user whose `request.auth.uid == specialistId`. No blanket `list` is permitted on `/specialists`.
2. **Pseudonymized Synthetic Records Only**: All records in `/followUps/{followUpId}` MUST satisfy `isSynthetic == true` and `syntheticCode` matching `^PAC-SYN-[0-9]{4}$` to enforce privacy-by-design and prevent storing real patient PII.
3. **Master Gate Relational Integrity**: Subcollection documents at `/followUps/{followUpId}/entries/{entryId}` cannot be read or written unless the parent `/followUps/{followUpId}` exists and belongs to `request.auth.uid`.
4. **Immutable Ownership & Temporal Integrity**: `specialistId`, `uid`, `syntheticCode`, and `createdAt` cannot be modified after creation; `createdAt` and `updatedAt` must equal `request.time`.
5. **Terminal State Protection**: Once a follow-up reaches `status == 'Retenido'`, updates via standard workflow are locked against silent status tampering unless explicitly reopened via the allowed reactivation action or prevented from arbitrary overwrites. (Specifically, `Retenido` is treated as terminal for clinical intervention entries until a dedicated update rule or delete/archive, or we lock `Retenido` from further modification per Terminal State Locking).

## 2. "Dirty Dozen" Adversarial Payloads (Expected: PERMISSION_DENIED)
1. **Unauthenticated Read/Write**: Reading `/followUps` with `auth == null`.
2. **Unverified Email Write**: Creating a follow-up with `email_verified: false`.
3. **Cross-Tenant Read**: Specialist B attempting `get` on Specialist A's `/specialists/{A}` or `/followUps/{docA}`.
4. **Blanket List Scraping**: Listing `/followUps` without `where('specialistId', '==', uid)`.
5. **Identity Spoofing on Create**: Specialist A creating `/followUps/{id}` with `specialistId: 'uid_B'`.
6. **Privilege Escalation on Profile**: Specialist creating `/specialists/{uid}` with `role: 'admin'`.
7. **Shadow Field Injection**: Creating/updating a follow-up with an extra undeclared field `{ ..., isVerifiedOverride: true }`.
8. **ID Poisoning**: Creating a document with ID exceeding 128 chars or containing invalid characters (`../bad$id`).
9. **Real PII / Non-Synthetic Bypass**: Creating a follow-up with `isSynthetic: false` or `syntheticCode: "Juan Perez"`.
10. **Client Timestamp Tampering**: Creating a follow-up with forged `createdAt` not equal to `request.time`.
11. **Orphan Subcollection Write**: Creating `/followUps/nonexistent/entries/e1`.
12. **Terminal State Overwrite**: Updating a follow-up whose existing `status` is `'Retenido'`.
