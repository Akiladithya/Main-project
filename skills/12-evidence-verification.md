# Evidence Verification Skill

## Responsibility
Check whether generated claims are supported by retrieved lecture evidence.

## Workflow

```text
Generated Answer
 -> Claim Extraction
 -> Evidence Matching
 -> Verification
 -> Accept / Repair
```

## Claim Example

```text
C1: TCP uses a three-way handshake.
C2: Client sends SYN.
C3: Server sends SYN-ACK.
C4: Client sends ACK.
```

## Evidence Sources
Verify claims against:
- timestamped speech
- visual descriptions
- visual events

## Outcomes

### Supported
Keep the claim.

### Unsupported
Do not silently accept it.

Trigger:
```text
Additional Retrieval
 -> Expanded Evidence
 -> Regeneration
 -> Verification
```

## Design Principle
Verification is a separate responsibility from generation.

## Failure Policy
If a claim remains unsupported after reasonable retrieval:
- remove it
- qualify it
- or state that the available lecture evidence is insufficient

Never replace missing evidence with model knowledge without explicitly distinguishing it.
