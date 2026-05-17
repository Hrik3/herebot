# Template Abbreviations

Hardcoded response templates use the `TPL.` prefix. Text without this prefix is learned, retrieved, user-provided, or language-model generated.

## Ask

- `TPL.ASK.ALIAS` - ask whether an alias should be used for self-reference.
- `TPL.ASK.ANALOGY` - ask whether two sentences are related.
- `TPL.ASK.CAT` - ask whether a subject is a category/type of an object.
- `TPL.ASK.CLARIFY` - ask for clarification.
- `TPL.ASK.CONN.REL` - ask whether a connector relation was understood.
- `TPL.ASK.CONN.ROLE` - ask what role a connector has.
- `TPL.ASK.CONSCIENCE` - ask whether a conscience rule is appropriate.
- `TPL.ASK.DEF` - ask for a word or phrase definition.
- `TPL.ASK.DIFF.LINK` - ask whether a shadow differentiation link should be committed.
- `TPL.ASK.FB` - ask for positive/negative/neutral feedback.
- `TPL.ASK.ISA` - ask whether subject is an object.
- `TPL.ASK.LOGIC` - ask for confirmation of a generated logic rule.
- `TPL.ASK.MEAN` - ask what a known token means in context.
- `TPL.ASK.MEANS` - ask whether one item means another.
- `TPL.ASK.PATTERN.LABEL` - ask for a label for a detected pattern.
- `TPL.ASK.PRONOUN` - ask whether a pronoun maps to the engine.
- `TPL.ASK.REL` - ask whether two items are related.
- `TPL.ASK.RULE` - ask for confirmation of a behavioral rule.
- `TPL.ASK.SENT` - ask for sentence meaning.
- `TPL.ASK.SYN` - ask whether two tokens are synonyms/similar.
- `TPL.ASK.TEACH` - ask for step-by-step teaching.
- `TPL.ASK.WORD.CONFLICT` - ask how to resolve conflicting word labels.

## Knowledge

- `TPL.DEF` - known definition statement.
- `TPL.DEF.EQ` - definition equality statement.
- `TPL.DEF.KNOWN` - known word definition.
- `TPL.DEF.LIKE` - definition similarity statement.
- `TPL.DEF.MAYBE` - uncertain definition statement.
- `TPL.DEF.PHRASE` - known phrase definition.
- `TPL.DEF.RECALL` - recalled definition.
- `TPL.DEF.REL` - definition with relations.
- `TPL.HYP.GRAPH` - graph-neighbor hypothesis.
- `TPL.HYP.REL` - relation hypothesis.
- `TPL.HYP.SIM` - similarity hypothesis.
- `TPL.NO.CTX` - no context available.
- `TPL.NO.DEF` - no definition available.
- `TPL.OBS.FREQ` - observed frequent token.
- `TPL.OBS.HIGH` - high-frequency observation.
- `TPL.OBS.IMPORTANT` - possible important token.
- `TPL.PATTERN` - generated structural pattern.
- `TPL.RECALL.PHRASE` - recalled similar phrase.

## State And Mode

- `TPL.ACK.ACTION` - generated action acknowledgement.
- `TPL.ACK.LISTEN` - idle/listening acknowledgement.
- `TPL.CHAT.MORE` - generic chat continuation request.
- `TPL.CONV.CONT` - conversation continuation.
- `TPL.GUILT.ACTIVE` - guilt signal active.
- `TPL.GUILT.CHECK` - uncertainty about prior action.
- `TPL.GUILT.CONFLICT` - conflicted response signal.
- `TPL.GUILT.ERROR` - prior mistake signal.
- `TPL.GUILT.UNEASY` - unease signal.
- `TPL.GUILT.WRONG` - possible wrong action signal.
- `TPL.LEARN` - learning fallback.
- `TPL.MODE.CHAT.OFF` - chat mode disabled.
- `TPL.MODE.CHAT.ON` - chat mode enabled.
- `TPL.PFX.CURIOUS` - curiosity state prefix.
- `TPL.PFX.LOW_COMP` - low comprehension prefix.
- `TPL.PFX.LOW_ENERGY` - low energy prefix.
- `TPL.PFX.THREAT` - threat/uncertainty prefix.
- `TPL.PFX.THREAT.HIGH` - high threat prefix.
- `TPL.PFX.TRUST` - trust/confidence prefix.
- `TPL.STATE.ASSOC` - word-state association.
- `TPL.STATE.CONF` - confidence state.
- `TPL.STATE.CONFLICT.MENU` - state conflict menu.
- `TPL.STATE.CONFLICT.PENDING` - pending state conflict.
- `TPL.STATE.CONFLICT.STALE` - stale state conflict skipped.
- `TPL.STATE.CONTEXT` - current state context.
- `TPL.STATE.CURIOUS` - curiosity state.
- `TPL.STATE.CURRENT` - current state label.
- `TPL.STATE.DRAIN` - draining/low-resource state.
- `TPL.STATE.HARD` - difficulty state.
- `TPL.STATE.LOW_COMP` - low comprehension state.
- `TPL.STATE.LOW_ENERGY` - low energy state.
- `TPL.STATE.MAYBE_WRONG` - possible wrongness state.
- `TPL.STATE.NO_PARSE` - parsing failure state.
- `TPL.STATE.OK` - positive state.
- `TPL.STATE.PAUSE` - pause-needed state.
- `TPL.STATE.THREAT` - uncertainty/threat state.
- `TPL.STATE.TOPIC` - state-linked topic.
- `TPL.STATE.UNDERSTAND` - understanding state.
- `TPL.STATE.UNSURE` - unsure state.
- `TPL.STATE.UPDATED` - generated state update acknowledgement.
- `TPL.TOPIC` - topic continuation.

## Math, Procedure, Idle

- `TPL.IDLE.EMPTY` - idle reflection with no memory.
- `TPL.IDLE.STATS` - idle reflection stats.
- `TPL.IDENTITY.ALIAS` - generated self-reference alias response.
- `TPL.MATH.RESULT` - direct math result.
- `TPL.MATH.SOLUTION` - solved equation/procedure result.
- `TPL.NGRAM` - n-gram continuation.
- `TPL.PROC.LEARN` - procedure learned.
