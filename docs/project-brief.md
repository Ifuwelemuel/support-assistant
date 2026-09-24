# support-assistant — project brief
## Problem
Support teams lose time triaging tickets and re-writing the same answers.
Routing is manual and inconsistent; answers vary by agent.
## What we are building
1.
**Ticket classifier** — routes each incoming ticket to a category.
Classical ML first (Week 2), then a fine-tuned transformer (Week 3).
2.
**RAG answerer** — drafts a grounded, cited reply from the support
knowledge base (Week 4 locally, Week 9 on Microsoft Foundry).
## v1 target
A ticket goes in; a category and a cited draft reply come out, served from
Azure, with training and deployment automated end-to-end (Weeks 6–8).
## Non-goals for now
- Multi-language tickets
- Live chat or voice
- Automatically sending replies without a human in the loop
## How we know it works
- Classifier beats the majority-class baseline by a stated margin (Week 2)
- Answerer scores above baseline on retrieval recall and faithfulness (Week 4)
- Zero manual steps between a push and a deployed model (Week 8)