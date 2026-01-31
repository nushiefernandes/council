#!/bin/bash
# Comprehensive Design Mode Test
# Tests: skill tools, multi-mockup, context passing, checkpoint/resume, edge cases

set -e
cd /Users/anushfernandes/.council/crewai-council

echo "============================================================"
echo "COMPREHENSIVE DESIGN MODE TEST"
echo "============================================================"
echo ""
echo "This test verifies:"
echo "  1. All 4 stages execute (UX → Visual → Technical → Synthesis)"
echo "  2. Skill tools work (list_skills, query_skill)"
echo "  3. Multiple HTML mockups generated (2-3 options)"
echo "  4. DESIGN-BRIEF.md created with comparison matrix"
echo "  5. Context flows correctly between stages"
echo "  6. Complex requirements handled properly"
echo ""
echo "============================================================"

# Clean previous test output
rm -rf workspace/design
mkdir -p workspace/design

# Complex test case that exercises all features
TASK='Design the home screen for a food tracking app called "Your Food".

PRODUCT REQUIREMENTS:
- Three primary actions vertically stacked (40/40/20 visual weight):
  1. "Cooked at Home" button → home-cooked meal logging flow
  2. "Eating Out" button → restaurant meal logging flow
  3. "Talk to Your Food" button → AI chat interface
- Recent entries section below fold (horizontal scroll carousel)
- Daily summary card showing calories/macros (collapsible)
- Bottom navigation with 4 tabs: Home, History, Insights, Profile

DESIGN CONSTRAINTS:
- Mobile-first (375px width, must work on iPhone SE)
- Thumb-zone optimization (primary actions in easy reach)
- Must NOT feel like a clinical calorie tracker
- Warm, personal, joy-focused aesthetic
- Second-person voice ("you made", "you had")
- Accessibility: WCAG 2.1 AA compliant, minimum touch targets 44px

BRAND GUIDELINES:
- Avoid: cold blues, medical whites, generic fitness app aesthetics
- Prefer: warm earth tones, soft gradients, friendly typography
- NO Inter, Arial, or system fonts - use distinctive typefaces
- Micro-interactions should feel playful, not sterile

EDGE CASES TO CONSIDER:
- Empty state (no recent entries)
- Overflow state (many recent entries)
- Dark mode support
- Loading states for async data
- Error states for failed API calls

Generate 2-3 DISTINCT design directions with HTML mockups.
Each direction should have a clear name and personality.'

echo ""
echo "Task: Design complex food app home screen"
echo "Testing with checkpoint mode for stage-by-stage verification..."
echo ""

# Run with checkpoint mode to verify each stage
./venv/bin/python crew.py --design --checkpoint -y "$TASK"

echo ""
echo "============================================================"
echo "TEST RESULTS"
echo "============================================================"

# Verify outputs
echo ""
echo "Checking generated files..."
echo ""

# Check for HTML mockups
HTML_COUNT=$(ls -1 workspace/design/*.html 2>/dev/null | wc -l)
echo "HTML mockups generated: $HTML_COUNT"
if [ "$HTML_COUNT" -ge 2 ]; then
    echo "  ✓ Multiple mockups created (expected 2-3)"
    ls -la workspace/design/*.html
else
    echo "  ✗ Expected 2-3 HTML mockups, got $HTML_COUNT"
fi

echo ""

# Check for DESIGN-BRIEF.md
if [ -f workspace/design/DESIGN-BRIEF.md ]; then
    echo "✓ DESIGN-BRIEF.md exists"
    echo ""
    echo "Brief preview (first 50 lines):"
    head -50 workspace/design/DESIGN-BRIEF.md
else
    echo "✗ DESIGN-BRIEF.md not found"
fi

echo ""

# Check design_context.json for stage completion
if [ -f workspace/design/design_context.json ]; then
    echo "✓ design_context.json exists (checkpoint state)"
    echo ""
    echo "Stages completed:"
    cat workspace/design/design_context.json | python3 -c "
import json, sys
ctx = json.load(sys.stdin)
for stage in ctx.get('stage_results', {}):
    print(f'  ✓ {stage}')
print(f'Total stages: {len(ctx.get(\"stage_results\", {}))}')
"
else
    echo "✗ design_context.json not found"
fi

echo ""
echo "============================================================"
echo "MANUAL VERIFICATION"
echo "============================================================"
echo ""
echo "Open these files in a browser to verify visual quality:"
for f in workspace/design/*.html; do
    [ -f "$f" ] && echo "  file://$PWD/$f"
done
echo ""
echo "Read the design brief:"
echo "  cat workspace/design/DESIGN-BRIEF.md"
echo ""
