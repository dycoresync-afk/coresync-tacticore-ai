# Contributing to CoreSync: TactiCore AI 🚀

Thank you for your interest in contributing to TactiCore AI! We're building an AI-powered mission planning platform for defense, law enforcement, and enterprise security, fusing real-time intel (HUMINT/SIGINT/OSINT) with explainable AI and Web3 for GDPR/ITAR compliance. Your contributions help us outsmart threats and secure lives.

We welcome contributions from everyone, especially those with experience in AI, security, or open-source. Whether it's code, docs, or feedback, follow these guidelines to get started. For major changes, please open an issue first.

## Code of Conduct
We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold it. Reports of abusive behavior go to dy.coresync@gmail.com.

## Getting Started
1. **Fork the Repo**: Click "Fork" on GitHub to create your copy.
2. **Clone Locally**: `git clone https://github.com/yourusername/coresync-tacticore-ai.git`
3. **Create a Branch**: `git checkout -b feature/your-feature-name` (e.g., `feature/data-fusion-enhance`).
4. **Set Up Environment**:
   - Install Python 3.10+.
   - Run `pip install -r requirements.txt`.
   - For Web3 stubs: Ensure Solana-py is installed (planned integration).
   - Test setup: `python src/main.py` (runs demo fusion).
5. **Explore Alpha**: Check `examples/demo_mission_planning.ipynb` for Jupyter-based mission sims.

## How to Contribute
We prioritize secure, compliant contributions—review our [compliance slide](docs/pitch-deck/CoreSync_AI-Powered_Mission_Planning.pdf) for GDPR/ITAR context. Start small:

- **Issues**: 
  - Search existing issues before creating new ones.
  - Use labels: "good-first-issue" for beginners (e.g., docs tweaks), "bug" for fixes, "enhancement" for features like zk-proofs.
  - For defsec-sensitive ideas, discuss privately via email first.
- **Documentation**: Improve README, API docs (`docs/api.md`), or roadmap (`docs/roadmap.md`). Add examples for OSINT fusion.
- **Code**: Focus on modular stubs in `src/tacticore/` (e.g., enhance `data_fusion.py` with real PyTorch models).
- **Tests**: Add to `tests/` using pytest. Run `pytest` before committing.
- **Other**: Report bugs, suggest UN/DoD integrations, or contribute to Web3 pilots (Q1 2026 milestone).

Look for "help-wanted" or "good-first-issue" labels—ideal for newcomers.

## Development Workflow
1. **Make Changes**: Edit in your branch. Follow style below.
2. **Test Locally**: 
   - Run tests: `pytest tests/`.
   - Lint: Use Black (`pip install black; black src/`).
   - Demo: `python src/main.py`.
3. **Commit**: Use descriptive messages, e.g., "feat: add OSINT fusion weights" or "fix: resolve XAI stub error".
4. **Push**: `git push origin feature/your-feature-name`.
5. **Pull Request (PR)**:
   - Open PR to `main` branch.
   - Title: "feat/fix/docs: Brief description".
   - Body: Reference issue (e.g., "Closes #42"), explain changes, and add screenshots if UI-related.
   - For AI/security: Include how it aligns with compliance (e.g., no sensitive data in code).
   - We'll review within 7 days—expect feedback on tests/coverage.

PRs must pass CI (via `.github/workflows/ci.yml`). No auto-merge for alpha stage.

## Code Style & Best Practices
- **Python**: PEP 8 compliant. Use Black for formatting, Flake8 for linting.
- **AI/ML**: Modular (e.g., separate fusion from XAI). Use SHAP for explanations; avoid black-box models.
- **Security**: No hard-coded secrets. For Web3 stubs, use testnets only. Flag ITAR-sensitive code.
- **Commits**: Conventional Commits (e.g., "feat:", "fix:", "docs:").
- **Branching**: Feature branches from `main`; rebase before PR.

Install tools: `pip install black flake8 pytest`.

## Reporting Security Issues
For vulnerabilities (e.g., in fusion or Web3 stubs), email dy.coresync@gmail.com privately—do not open public issues. We'll triage and disclose responsibly, aligning with our defsec focus.

## Questions?
- Join discussions on GitHub Issues or email dy.coresync@gmail.com.
- For UN/DoD context, reference our [pitch deck](docs/pitch-deck/CoreSync_AI-Powered_Mission_Planning.pdf).

Thanks for contributing! Your work advances secure mission planning. 🌟

*Inspired by open-source best practices from r/opensource and r/learnprogramming communities.*
