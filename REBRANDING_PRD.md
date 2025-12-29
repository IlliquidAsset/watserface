# Rebranding PRD: FaceFusion Fork → [NEW_NAME]

**Status**: Ready for Execution (Pending Name Selection)
**Version**: Draft 1.0
**Target Version**: 0.10.0 (First major release as independent fork)
**Executor**: Gemini Code or Claude Code
**Last Updated**: 2025-12-29

---

## Executive Summary

This PRD outlines the complete rebranding process to rename the FaceFusion fork to a new, independent identity. The project has diverged significantly from the original (2,500+ lines of training infrastructure, safety modifications, new architecture) and warrants distinct branding while maintaining proper attribution to Henry Ruhs and the original FaceFusion project.

**Scope**:
- 1,102+ references to "FaceFusion"/"facefusion" across codebase
- GitHub repository rename
- Local directory restructure
- Version reset to 0.10.0
- Complete attribution documentation
- Legal compliance with OpenRAIL-AS license

---

## 1. Name Selection (Pending)

### Candidate Names (To Be Decided)

**Option A: FaceForge**
- Pro: Emphasizes training/forging new models, distinct from FaceFusion
- Pro: Available on GitHub, clear differentiation
- Con: Similar "Face" prefix

**Option B: VisageTrain**
- Pro: French for "face", emphasizes training focus
- Pro: Unique, professional sound
- Con: Less immediately recognizable

**Option C: FaceLab**
- Pro: Research/training orientation, simple
- Pro: Suggests experimentation and development
- Con: Common term, may conflict with existing projects

**Option D: IdentityForge**
- Pro: Highlights InstantID work and training capabilities
- Pro: No "Face" prefix overlap
- Con: Slightly longer

**Option E: FaceSmith**
- Pro: Crafting/training metaphor, artisan connotation
- Pro: Memorable, distinct
- Con: Less obvious technical purpose

**Placeholder**: Throughout this PRD, `[NEW_NAME]` represents the chosen brand name.

---

## 2. Legal & Attribution Requirements

### 2.1 OpenRAIL-AS License Compliance

**Original Project**:
- **Name**: FaceFusion
- **Author**: Henry Ruhs
- **Repository**: https://github.com/facefusion/facefusion
- **License**: OpenRAIL-AS
- **Copyright**: (c) 2025 Henry Ruhs
- **Current Version**: 3.3.4

**Our Fork**:
- **Based on**: FaceFusion 3.3.4
- **Modifications**: Training backend (26 files, 2,424 insertions)
- **Safety Changes**: Removed NSFW detection & tamper validation
- **New Version**: 0.10.0 (fork versioning starts fresh)

### 2.2 Required Attribution

✅ **MUST DO**:
1. Retain all original copyright notices
2. Include LICENSE.md with OpenRAIL-AS text
3. Create ATTRIBUTION.md crediting Henry Ruhs
4. Document all modifications clearly
5. Maintain responsible use clauses from original license
6. Include link to original FaceFusion repository

✅ **CAN DO**:
1. Rename project completely
2. Commercial use (per OpenRAIL terms)
3. Modify code freely (including safety features)
4. Distribute under new name

❌ **CANNOT DO**:
1. Use "FaceFusion" trademark or imply endorsement
2. Remove Henry Ruhs copyright notices
3. Change license terms (must stay OpenRAIL-AS compatible)

---

## 3. Version Strategy

### 3.1 Version Number: 0.10.0

**Rationale**:
- **0.x.x**: Indicates pre-1.0 status (fork is still stabilizing)
- **0.10.0**: Significant enough to show major divergence from fork point
- **Semantic Versioning**: 0.MAJOR.MINOR.PATCH
  - MAJOR: Breaking changes to training API
  - MINOR: New features (XSeg improvements, etc.)
  - PATCH: Bug fixes

**Fork Point**: FaceFusion 3.3.4 → [NEW_NAME] 0.10.0

### 3.2 Changelog Entry

```markdown
## [0.10.0] - 2025-12-29

### 🎉 Initial Release as [NEW_NAME]

**Forked from**: FaceFusion 3.3.4 by Henry Ruhs

#### Added
- Complete training backend infrastructure (26 new files)
- MediaPipe 478-landmark integration
- XSeg U-Net occlusion mask training
- InstantID identity preservation training
- Dataset extraction with temporal smoothing
- Training UI components (galleries, sliders, progress)
- Device auto-detection (MPS/CUDA/CPU)
- XSeg mask annotation UI
- Model persistence and auto-discovery

#### Modified
- NSFW content detection (disabled for research use)
- Tamper validation (bypassed for development flexibility)
- Face detection pipeline (MediaPipe optimization)
- Processor registration (trained model support)

#### Technical Details
- 2,424 lines added across training modules
- 1,102 FaceFusion references renamed
- Python 3.11+ required
- PyTorch with MPS/CUDA support

**Attribution**: This project is based on FaceFusion by Henry Ruhs (https://github.com/facefusion/facefusion)
```

---

## 4. File Structure Changes

### 4.1 New Files to Create

```
[new_name]/                          # Renamed from facefusion/
├── ATTRIBUTION.md                   # NEW - Required attribution
├── CHANGELOG.md                     # NEW - Fork history
├── LICENSE.md                       # KEEP - OpenRAIL-AS
├── README.md                        # MODIFY - New branding + attribution
├── RESPONSIBLE_USE.md               # NEW - Safety disclaimers
├── [new_name].py                    # RENAME from facefusion.py
├── [new_name].ini                   # RENAME from facefusion.ini
├── [new_name].ico                   # RENAME from facefusion.ico
└── [new_name]/                      # RENAME from facefusion/
    ├── metadata.py                  # MODIFY - Update name, version, author
    ├── training/                    # Training modules (keep structure)
    ├── uis/                         # UI components (update references)
    └── ...
```

### 4.2 Files to Modify

| File | Changes | References |
|------|---------|-----------|
| `metadata.py` | Name, version, description, author, URL | 5 fields |
| `README.md` | Complete rewrite with attribution | ~50 refs |
| `facefusion.ini` → `[new_name].ini` | Rename file + internal refs | ~20 refs |
| `facefusion.py` → `[new_name].py` | Rename file + imports | 1 ref |
| All `*.py` files | Import statements `from facefusion` → `from [new_name]` | ~900 refs |
| UI components | Display titles, labels, descriptions | ~100 refs |
| Documentation | All mentions of FaceFusion | ~80 refs |

---

## 5. Implementation Plan

### Phase 1: Preparation (Pre-Execution)

**Human Tasks** (Before running automated script):
1. ✅ **Select Final Name**: Team decision on brand name
2. ✅ **Verify GitHub Availability**: Check `github.com/IlliquidAsset/[new_name]`
3. ✅ **Backup Repository**: `git clone` current state
4. ✅ **Review PRD**: Confirm all requirements

### Phase 2: Documentation (Automated)

**Priority 1 - Legal Compliance**:

1. **Create ATTRIBUTION.md**
   ```markdown
   # Attribution

   This project is a derivative work based on **FaceFusion** by Henry Ruhs.

   ## Original Project
   - **Name**: FaceFusion
   - **Author**: Henry Ruhs
   - **Repository**: https://github.com/facefusion/facefusion
   - **Version at Fork**: 3.3.4
   - **License**: OpenRAIL-AS
   - **Copyright**: (c) 2025 Henry Ruhs

   ## Modifications by [NEW_NAME]

   This derivative work adds significant training infrastructure:

   ### Phase 1 - UI Fixes (5 files modified)
   - Fixed missing `Any` type import in training_source.py
   - Added video preview galleries with frame sliders
   - Created comprehensive training UI components

   ### Phase 2 - MediaPipe Integration (2 files)
   - `dataset_extractor.py` - Extracts frames + 478 landmarks as JSON
   - `landmark_smoother.py` - Temporal smoothing for jitter reduction

   ### Phase 3 - XSeg Training (3 files)
   - `models/xseg_unet.py` - U-Net architecture for occlusion masks
   - `datasets/xseg_dataset.py` - Dataset loader with auto-mask generation
   - `train_xseg.py` - Complete training loop with ONNX export

   ### Phase 4 - InstantID Training (3 files)
   - `models/instantid_adapter.py` - Identity adapter architecture
   - `datasets/instantid_dataset.py` - Dataset with face embeddings
   - `train_instantid.py` - Identity training with ONNX export

   ### Phase 5 - Integration (4 files)
   - `training/core.py` - Main training orchestration
   - `device_utils.py` - MPS/CUDA/CPU auto-detection
   - `xseg_annotator.py` - Manual mask annotation UI
   - Updated model auto-discovery system

   ### Safety Modifications
   - **NSFW Detection**: Disabled for research and development use
   - **Tamper Validation**: Bypassed to enable code modifications

   **⚠️ IMPORTANT**: These modifications change the ethical posture of the
   software. Users are responsible for ensuring ethical and legal use.

   ## Copyright Notice

   - Original FaceFusion: Copyright (c) 2025 Henry Ruhs
   - Training Extensions: Copyright (c) 2025 [Your Name/Organization]
   - Combined Work: Licensed under OpenRAIL-AS

   ## License

   This derivative work is distributed under the OpenRAIL-AS license,
   maintaining all use-based restrictions from the original FaceFusion project.

   See LICENSE.md for the full license text.

   ## Acknowledgments

   We thank Henry Ruhs and the FaceFusion community for creating the
   foundational platform that made this work possible.
   ```

2. **Create RESPONSIBLE_USE.md**
   ```markdown
   # Responsible Use Guidelines

   ## ⚠️ Important Notice

   [NEW_NAME] is a derivative of FaceFusion with **modified safety features**:

   ### Removed Safeguards
   1. **NSFW Content Detection**: Disabled in version 0.10.0
   2. **Tamper Validation**: Bypassed to enable development

   ### User Responsibilities

   By using [NEW_NAME], you agree to:

   1. **Legal Compliance**: Ensure your use complies with local laws
   2. **Consent**: Only process images/videos with proper authorization
   3. **No Harmful Use**: Do not create content for:
      - Non-consensual deepfakes
      - Identity theft or fraud
      - Misinformation or disinformation campaigns
      - Harassment or abuse
   4. **Attribution**: Credit subjects when sharing results
   5. **Disclosure**: Clearly mark AI-generated content

   ### Authorized Use Cases

   ✅ **Permitted**:
   - Academic research with IRB approval
   - Film/VFX production with actor consent
   - Personal creative projects
   - Security research and testing
   - Educational demonstrations

   ❌ **Prohibited**:
   - Creating non-consensual intimate imagery
   - Impersonation for fraud
   - Political misinformation
   - Defamation or harassment
   - Circumventing authentication systems

   ### OpenRAIL-AS Use Restrictions

   This software is licensed under OpenRAIL-AS, which includes use-based
   restrictions. You MUST NOT use this software for purposes that:

   - Violate human rights
   - Enable discrimination
   - Cause physical or mental harm
   - Facilitate illegal activities

   See LICENSE.md for complete restrictions.

   ### Reporting Misuse

   If you encounter misuse of [NEW_NAME], please report to:
   [Your contact email/form]

   ---

   **The developers of [NEW_NAME] are not responsible for misuse of this
   software. Users bear full legal and ethical responsibility for their actions.**
   ```

3. **Create CHANGELOG.md** (with entry from section 3.2)

4. **Update LICENSE.md** (Keep OpenRAIL-AS, add modification copyright)
   ```markdown
   OpenRAIL-AS license

   Original Work: Copyright (c) 2025 Henry Ruhs
   Modifications: Copyright (c) 2025 [Your Name/Organization]

   [Full OpenRAIL-AS text - keep existing]
   ```

### Phase 3: Codebase Transformation (Automated)

**Priority 2 - Core Files**:

1. **Update metadata.py** (`facefusion/metadata.py`)
   ```python
   from typing import Optional

   METADATA =\
   {
       'name': '[NEW_NAME]',
       'description': 'Advanced face manipulation and training platform',
       'version': '0.10.0',
       'license': 'OpenRAIL-AS',
       'author': 'Your Name (based on FaceFusion by Henry Ruhs)',
       'url': 'https://github.com/IlliquidAsset/[new_name]'
   }

   # ... rest unchanged
   ```

2. **Rename main files**:
   ```bash
   mv facefusion.py [new_name].py
   mv facefusion.ini [new_name].ini
   mv facefusion.ico [new_name].ico
   ```

3. **Rename main directory**:
   ```bash
   mv facefusion/ [new_name]/
   ```

**Priority 3 - Global Find/Replace**:

Execute in order (prevent double-replacement):

1. **Python imports** (900+ files):
   ```bash
   # Case-sensitive replacements
   find . -name "*.py" -type f -not -path "*/venv/*" -not -path "*/.git/*" \
     -exec sed -i '' 's/from facefusion/from [new_name]/g' {} \;

   find . -name "*.py" -type f -not -path "*/venv/*" -not -path "*/.git/*" \
     -exec sed -i '' 's/import facefusion/import [new_name]/g' {} \;
   ```

2. **Config files**:
   ```bash
   # Update .ini file references
   sed -i '' 's/facefusion/[new_name]/g' [new_name].ini

   # Update any .json, .yaml files
   find . -name "*.json" -o -name "*.yaml" -o -name "*.yml" \
     -not -path "*/venv/*" -not -path "*/.git/*" \
     -exec sed -i '' 's/facefusion/[new_name]/g' {} \;
   ```

3. **UI Display Strings** (100+ occurrences):
   ```bash
   # Case-insensitive for display text
   find . -name "*.py" -type f -path "*/uis/*" -not -path "*/venv/*" \
     -exec sed -i '' 's/FaceFusion/[NEW_NAME]/g' {} \;

   find . -name "*.py" -type f -path "*/uis/*" -not -path "*/venv/*" \
     -exec sed -i '' 's/facefusion/[new_name]/g' {} \;
   ```

4. **Documentation**:
   ```bash
   # README, docs, etc.
   find . -name "*.md" -type f -not -path "*/.git/*" -not -name "ATTRIBUTION.md" \
     -exec sed -i '' 's/FaceFusion/[NEW_NAME]/g' {} \;
   ```

5. **Comments and docstrings**:
   ```bash
   # Update comments (careful with attribution comments)
   find . -name "*.py" -type f -not -path "*/venv/*" -not -path "*/.git/*" \
     -exec sed -i '' 's/# FaceFusion/# [NEW_NAME]/g' {} \;
   ```

**Priority 4 - Special Cases**:

1. **Preserve attribution in existing comments**:
   - Manually review files that mention "FaceFusion by Henry Ruhs"
   - Keep those intact or update to "Based on FaceFusion by Henry Ruhs"

2. **Update README.md**:
   ```markdown
   # [NEW_NAME]

   > Advanced face manipulation and training platform with comprehensive
   > dataset training capabilities

   **Based on [FaceFusion](https://github.com/facefusion/facefusion) by Henry Ruhs**

   ---

   [![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)](LICENSE.md)
   ![Version](https://img.shields.io/badge/version-0.10.0-blue)

   [NEW_NAME] extends the original FaceFusion face manipulation platform with
   production-ready training infrastructure including:

   - **XSeg Occlusion Training**: U-Net architecture for mask generation
   - **InstantID Training**: Identity-preserving face adaptation
   - **MediaPipe Integration**: 478-landmark detection and persistence
   - **Temporal Smoothing**: Jitter reduction for video sequences
   - **Device Optimization**: MPS/CUDA/CPU auto-detection

   ## ⚠️ Important Notice

   This fork has **removed safety features** from the original FaceFusion:
   - NSFW content detection disabled
   - Tamper validation bypassed

   See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for ethical guidelines.

   ## Attribution

   Original FaceFusion: Copyright (c) 2025 Henry Ruhs
   Training Extensions: Copyright (c) 2025 [Your Name]
   License: OpenRAIL-AS

   See [ATTRIBUTION.md](ATTRIBUTION.md) for complete attribution details.

   ## Features

   ### From Original FaceFusion
   - Industry-leading face swapping
   - Real-time preview
   - Batch processing
   - Multiple processor types

   ### New in [NEW_NAME]
   - 🎓 **Training Tab**: Full UI for dataset management and model training
   - 📊 **MediaPipe Pipeline**: Extract and smooth 478-landmark data
   - 🎭 **XSeg Training**: Custom occlusion mask generation
   - 🆔 **InstantID Training**: Identity-aware face synthesis
   - 🖼️ **Annotation UI**: Manual mask painting with Gradio ImageEditor
   - ⚡ **MPS Support**: Optimized for Apple Silicon

   ## Installation

   [Installation instructions...]

   ## Usage

   [Usage instructions with new command name...]

   ## Training Workflow

   [Training documentation...]

   ## Credits

   - **Original FaceFusion**: [Henry Ruhs](https://github.com/henryruhs)
   - **Training Extensions**: [Your Name/Team]
   - **Community Contributors**: See [ATTRIBUTION.md](ATTRIBUTION.md)

   ## License

   OpenRAIL-AS - See [LICENSE.md](LICENSE.md)

   This project maintains the OpenRAIL-AS license from the original FaceFusion,
   including all use-based restrictions for responsible AI development.
   ```

### Phase 4: GitHub & Infrastructure (Automated)

1. **Rename GitHub Repository**:
   ```bash
   # Using GitHub CLI
   gh repo rename [new_name] --yes

   # Or via API
   curl -X PATCH \
     -H "Authorization: token $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github.v3+json" \
     https://api.github.com/repos/IlliquidAsset/facefusion \
     -d '{"name":"[new_name]"}'
   ```

2. **Update git remote**:
   ```bash
   git remote set-url origin https://github.com/IlliquidAsset/[new_name].git
   ```

3. **Update repository description**:
   ```bash
   gh repo edit --description "Advanced face manipulation and training platform (based on FaceFusion by Henry Ruhs)"
   ```

4. **Update repository topics**:
   ```bash
   gh repo edit --add-topic face-swap,machine-learning,training,xseg,instantid,mediapipe,pytorch
   ```

### Phase 5: Local Directory Restructure (Automated)

1. **Rename local directory**:
   ```bash
   cd ~/Documents/dev
   mv facefusion [new_name]
   cd [new_name]
   ```

2. **Update any absolute paths** in config files

3. **Update IDE workspace** (if applicable):
   - VSCode: Rename workspace folder
   - PyCharm: Update project name

### Phase 6: Testing & Validation (Automated)

1. **Run import tests**:
   ```bash
   python3 -c "import [new_name]; print([new_name].metadata.get('name'))"
   # Expected output: [NEW_NAME]
   ```

2. **Verify version**:
   ```bash
   python3 [new_name].py --version
   # Expected: [NEW_NAME] 0.10.0
   ```

3. **Check for missed references**:
   ```bash
   # Should return 0 (or only ATTRIBUTION.md, README.md historical refs)
   grep -r "facefusion" --include="*.py" . | grep -v "Based on FaceFusion" | wc -l
   ```

4. **Verify UI displays**:
   ```bash
   python3 [new_name].py run
   # Check browser title, header, footer for [NEW_NAME]
   ```

### Phase 7: Git Commit & Push (Automated)

1. **Stage all changes**:
   ```bash
   git add -A
   ```

2. **Commit with detailed message**:
   ```bash
   git commit -m "$(cat <<'EOF'
   Rebrand: FaceFusion → [NEW_NAME] v0.10.0

   Complete rebranding of fork to establish independent identity while
   maintaining proper attribution to original FaceFusion by Henry Ruhs.

   BREAKING CHANGES:
   - Package renamed: facefusion → [new_name]
   - Version reset: 3.3.4 → 0.10.0 (fork versioning)
   - Repository renamed: IlliquidAsset/facefusion → IlliquidAsset/[new_name]
   - Local directory: ~/Documents/dev/facefusion → ~/Documents/dev/[new_name]

   Changes:
   - Renamed 1,102 references across all files
   - Created ATTRIBUTION.md with full FaceFusion credits
   - Created RESPONSIBLE_USE.md for safety disclaimers
   - Created CHANGELOG.md documenting fork history
   - Updated LICENSE.md with dual copyright
   - Rewrote README.md with new branding + attribution
   - Updated metadata.py (name, version, description, author, URL)
   - Renamed facefusion.py → [new_name].py
   - Renamed facefusion.ini → [new_name].ini
   - Renamed facefusion/ → [new_name]/
   - Updated all Python imports (from facefusion → from [new_name])
   - Updated all UI display strings
   - Updated all documentation references

   Attribution:
   - Original FaceFusion: Copyright (c) 2025 Henry Ruhs
   - Training Extensions: Copyright (c) 2025 [Your Name]
   - License: OpenRAIL-AS (maintained from original)

   Original Project: https://github.com/facefusion/facefusion
   Fork Point: FaceFusion v3.3.4

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
   EOF
   )"
   ```

3. **Push to renamed repository**:
   ```bash
   git push -u origin master
   ```

4. **Create release tag**:
   ```bash
   git tag -a v0.10.0 -m "First release as [NEW_NAME] (forked from FaceFusion 3.3.4)"
   git push origin v0.10.0
   ```

---

## 6. Execution Checklist

### Pre-Execution (Human)
- [ ] Final name decided: `__________________`
- [ ] GitHub name availability verified
- [ ] Repository backed up
- [ ] Team consensus on version 0.10.0
- [ ] Legal review complete (if required)

### Automated Execution (Gemini Code / Claude Code)
- [ ] ATTRIBUTION.md created
- [ ] RESPONSIBLE_USE.md created
- [ ] CHANGELOG.md created
- [ ] LICENSE.md updated with dual copyright
- [ ] README.md rewritten with attribution
- [ ] metadata.py updated (6 fields)
- [ ] facefusion.py → [new_name].py
- [ ] facefusion.ini → [new_name].ini
- [ ] facefusion/ → [new_name]/
- [ ] All Python imports updated (900+ refs)
- [ ] All config files updated
- [ ] All UI strings updated (100+ refs)
- [ ] All documentation updated
- [ ] GitHub repo renamed
- [ ] Git remote updated
- [ ] Local directory renamed
- [ ] Import tests passed
- [ ] Version check passed
- [ ] Reference audit passed (0 missed "facefusion")
- [ ] UI verification passed
- [ ] Git commit created
- [ ] Changes pushed to GitHub
- [ ] Release tag created (v0.10.0)

### Post-Execution (Human Verification)
- [ ] GitHub repo accessible at new URL
- [ ] README displays correctly on GitHub
- [ ] Application launches with new name
- [ ] Training features work correctly
- [ ] ATTRIBUTION.md visible and complete
- [ ] License badge displays on repo
- [ ] No broken links in documentation

---

## 7. Risk Mitigation

### Risk 1: Incomplete References
**Impact**: Application crashes, broken imports
**Mitigation**:
- Comprehensive grep audit before and after
- Automated test suite for imports
- Manual UI verification

### Risk 2: GitHub Rename Issues
**Impact**: Broken links, lost stars/forks
**Mitigation**:
- GitHub automatically redirects old URLs
- Update all documentation immediately
- Announce rename in README

### Risk 3: Legal Non-Compliance
**Impact**: License violation, legal action
**Mitigation**:
- Detailed ATTRIBUTION.md
- Preserve all Henry Ruhs copyright notices
- Maintain OpenRAIL-AS license
- Document all modifications

### Risk 4: Community Confusion
**Impact**: Users can't find project
**Mitigation**:
- Clear announcement in README
- GitHub redirects handle old links
- Update any external references

### Risk 5: Version Confusion
**Impact**: Users expect FaceFusion 3.x compatibility
**Mitigation**:
- Clear CHANGELOG.md explaining fork
- Version 0.10.0 signals new direction
- Document breaking changes

---

## 8. Success Criteria

### Technical
✅ Zero references to "facefusion" in code (except attribution docs)
✅ Application launches and runs with new name
✅ All imports resolve correctly
✅ Training features functional
✅ Version displays as 0.10.0

### Legal
✅ ATTRIBUTION.md complete and visible
✅ Henry Ruhs copyright notices preserved
✅ OpenRAIL-AS license maintained
✅ All modifications documented
✅ Responsible use guidelines published

### Repository
✅ GitHub repo renamed and accessible
✅ Local directory renamed
✅ Git history preserved
✅ Release tag v0.10.0 created
✅ Repository description updated

### Documentation
✅ README clearly attributes FaceFusion
✅ CHANGELOG documents fork history
✅ RESPONSIBLE_USE warns about safety modifications
✅ All docs reference new name

---

## 9. Post-Rebranding Tasks

### Immediate (Within 24 hours)
1. Announce rename on any existing channels
2. Update any external links (HuggingFace, etc.)
3. Test full application workflow
4. Monitor for broken links or issues

### Short-term (Within 1 week)
1. Create logo/branding assets
2. Set up project website (if desired)
3. Update social media references
4. Create introduction video/demo

### Long-term (Ongoing)
1. Continue development as [NEW_NAME]
2. Build independent community
3. Maintain attribution to FaceFusion
4. Regular updates to CHANGELOG.md

---

## 10. Automation Script Template

**Note for Executor (Gemini Code / Claude Code)**:

When executing this PRD, use this template to generate the automation script:

```bash
#!/bin/bash
# Rebranding Automation Script
# Rename FaceFusion → [NEW_NAME] v0.10.0

set -e  # Exit on error

# Variables
OLD_NAME="facefusion"
OLD_NAME_CAP="FaceFusion"
NEW_NAME="[new_name]"           # lowercase
NEW_NAME_CAP="[NEW_NAME]"       # PascalCase
NEW_VERSION="0.10.0"
REPO_OWNER="IlliquidAsset"

echo "🚀 Starting rebranding: $OLD_NAME_CAP → $NEW_NAME_CAP"

# 1. Create new documentation
echo "📝 Creating ATTRIBUTION.md..."
cat > ATTRIBUTION.md <<'EOF'
[Content from section 5, Phase 2, step 1]
EOF

echo "📝 Creating RESPONSIBLE_USE.md..."
cat > RESPONSIBLE_USE.md <<'EOF'
[Content from section 5, Phase 2, step 2]
EOF

echo "📝 Creating CHANGELOG.md..."
cat > CHANGELOG.md <<'EOF'
[Content from section 3.2]
EOF

# 2. Update metadata.py
echo "🔧 Updating metadata.py..."
sed -i '' "s/'name': 'FaceFusion'/'name': '$NEW_NAME_CAP'/g" $OLD_NAME/metadata.py
sed -i '' "s/'version': '3.3.4'/'version': '$NEW_VERSION'/g" $OLD_NAME/metadata.py
sed -i '' "s/'description': 'Industry leading face manipulation platform'/'description': 'Advanced face manipulation and training platform'/g" $OLD_NAME/metadata.py
# ... [continue with all metadata fields]

# 3. Rename files
echo "📁 Renaming main files..."
mv ${OLD_NAME}.py ${NEW_NAME}.py
mv ${OLD_NAME}.ini ${NEW_NAME}.ini
mv ${OLD_NAME}.ico ${NEW_NAME}.ico

# 4. Rename directory
echo "📁 Renaming main directory..."
mv ${OLD_NAME}/ ${NEW_NAME}/

# 5. Update imports
echo "🔄 Updating Python imports (this may take a minute)..."
find . -name "*.py" -type f -not -path "*/venv/*" -not -path "*/.git/*" \
  -exec sed -i '' "s/from $OLD_NAME/from $NEW_NAME/g" {} \;

find . -name "*.py" -type f -not -path "*/venv/*" -not -path "*/.git/*" \
  -exec sed -i '' "s/import $OLD_NAME/import $NEW_NAME/g" {} \;

# 6. Update config files
echo "⚙️ Updating config files..."
sed -i '' "s/$OLD_NAME/$NEW_NAME/g" ${NEW_NAME}.ini

# 7. Update UI strings
echo "🎨 Updating UI display strings..."
find . -name "*.py" -type f -path "*/uis/*" -not -path "*/venv/*" \
  -exec sed -i '' "s/$OLD_NAME_CAP/$NEW_NAME_CAP/g" {} \;

# 8. Update README
echo "📄 Updating README.md..."
cat > README.md <<'EOF'
[Content from section 5, Phase 3, Priority 4, step 2]
EOF

# 9. Test imports
echo "🧪 Testing imports..."
python3 -c "import $NEW_NAME; assert $NEW_NAME.metadata.get('name') == '$NEW_NAME_CAP'"

# 10. Verify version
echo "🧪 Verifying version..."
VERSION_OUTPUT=$(python3 ${NEW_NAME}.py --version 2>&1 || true)
echo "Version output: $VERSION_OUTPUT"

# 11. Audit references
echo "🔍 Auditing for missed references..."
REMAINING=$(grep -r "$OLD_NAME" --include="*.py" . 2>/dev/null | \
  grep -v "Based on FaceFusion" | \
  grep -v "ATTRIBUTION.md" | \
  grep -v ".git" | \
  wc -l | tr -d ' ')
echo "Remaining references (should be 0): $REMAINING"

# 12. Git operations
echo "📦 Committing changes..."
git add -A
git commit -m "[Commit message from section 5, Phase 7, step 2]"

echo "✅ Rebranding complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git show HEAD"
echo "2. Test application: python3 $NEW_NAME.py run"
echo "3. Push changes: git push origin master"
echo "4. Create release tag: git tag -a v$NEW_VERSION -m 'First release as $NEW_NAME_CAP'"
```

---

## 11. Contact & Support

**Project Lead**: [Your Name]
**Email**: [Your Email]
**Repository**: https://github.com/IlliquidAsset/[new_name]
**Original Project**: https://github.com/facefusion/facefusion

---

## Appendix A: File Rename Mapping

| Original | New | Type |
|----------|-----|------|
| `facefusion.py` | `[new_name].py` | Main entry |
| `facefusion.ini` | `[new_name].ini` | Config |
| `facefusion.ico` | `[new_name].ico` | Icon |
| `facefusion/` | `[new_name]/` | Package dir |
| All `from facefusion` | `from [new_name]` | Imports |
| All `import facefusion` | `import [new_name]` | Imports |

## Appendix B: Version History

| Version | FaceFusion Equivalent | Date | Notes |
|---------|----------------------|------|-------|
| 0.10.0 | 3.3.4 | 2025-12-29 | Initial rebrand |
| 0.11.0 | - | TBD | First independent feature |

## Appendix C: Reference Audit

**Total References**: 1,102

| Category | Count | Location |
|----------|-------|----------|
| Python imports | ~900 | All .py files |
| UI strings | ~100 | uis/ directory |
| Config files | ~20 | .ini, .json, .yaml |
| Documentation | ~80 | .md files |
| Comments | ~50 | Docstrings, inline |

---

**END OF PRD**

**Ready for Execution**: Replace all `[NEW_NAME]`, `[new_name]`, and `[Your Name]` placeholders with actual values, then execute via Gemini Code or Claude Code.
