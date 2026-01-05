"""
Debug Bad Match Example.

This script demonstrates how to use X-Ray to debug a bad recommendation result.
It runs the competitor selection pipeline in "bad match" mode, captures the run_id,
and then queries the X-Ray API to generate a human-readable debug report.

The report shows:
    - What candidates were considered at each step
    - Why candidates were rejected
    - Where the logic went wrong
    - Confidence scores and reasoning factors
"""

import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, "c:\\Company\\xray_system")

from xray_sdk import XRayClient, Run, Step


# =============================================================================
# Debug Report Generator
# =============================================================================

class DebugReportGenerator:
    """Generates human-readable debug reports from X-Ray run data."""
    
    def __init__(self, run: Run):
        self.run = run
        self.issues_found: List[str] = []
        self.warnings: List[str] = []
    
    def generate_report(self) -> str:
        """Generate a comprehensive debug report."""
        lines = []
        
        # Header
        lines.extend(self._generate_header())
        lines.append("")
        
        # Run Summary
        lines.extend(self._generate_run_summary())
        lines.append("")
        
        # Step-by-step analysis
        lines.extend(self._generate_step_analysis())
        lines.append("")
        
        # Issues and Recommendations
        lines.extend(self._generate_issues_summary())
        
        return "\n".join(lines)
    
    def _generate_header(self) -> List[str]:
        """Generate report header."""
        width = 80
        lines = [
            "=" * width,
            " X-RAY DEBUG REPORT ".center(width, "="),
            "=" * width,
            "",
            f"Run ID:     {self.run.run_id}",
            f"Run Name:   {self.run.name}",
            f"Status:     {self.run.status.value}",
            f"Created:    {self.run.created_at}",
        ]
        
        if self.run.context:
            if self.run.context.user_id:
                lines.append(f"User ID:    {self.run.context.user_id}")
            if self.run.context.session_id:
                lines.append(f"Session ID: {self.run.context.session_id}")
            
            # Check for bad_match_mode in custom context
            if self.run.context.custom:
                bad_match = self.run.context.custom.get("bad_match_mode", False)
                if bad_match:
                    lines.append(f"⚠️  BAD MATCH MODE: ENABLED (intentional bad scoring)")
                    self.warnings.append("bad_match_mode is enabled - scoring is intentionally inverted")
        
        return lines
    
    def _generate_run_summary(self) -> List[str]:
        """Generate run summary section."""
        lines = [
            "",
            "─" * 80,
            " RUN SUMMARY ".center(80, "─"),
            "─" * 80,
        ]
        
        lines.append(f"Total Steps: {len(self.run.steps)}")
        
        if self.run.final_output:
            lines.append(f"Final Success: {self.run.final_output.success}")
            if self.run.final_output.result_ids:
                lines.append(f"Result IDs: {', '.join(self.run.final_output.result_ids)}")
            if self.run.final_output.error:
                lines.append(f"Error: {self.run.final_output.error}")
        
        if self.run.summary:
            lines.append(f"Candidates Considered: {self.run.summary.total_candidates_considered}")
            lines.append(f"Candidates Filtered: {self.run.summary.total_candidates_filtered}")
            lines.append(f"Filters Applied: {self.run.summary.total_filters_applied}")
        
        # Decision flow overview
        lines.append("")
        lines.append("Decision Flow:")
        for i, step in enumerate(self.run.steps):
            output_count = step.outputs.count if step.outputs else "?"
            arrow = "  └──" if i == len(self.run.steps) - 1 else "  ├──"
            status_icon = self._get_status_icon(step)
            lines.append(f"{arrow} Step {i+1}: {step.name} [{step.step_type}] → {output_count} outputs {status_icon}")
        
        return lines
    
    def _generate_step_analysis(self) -> List[str]:
        """Generate detailed step-by-step analysis."""
        lines = [
            "",
            "═" * 80,
            " STEP-BY-STEP ANALYSIS ".center(80, "═"),
            "═" * 80,
        ]
        
        for i, step in enumerate(self.run.steps):
            lines.extend(self._analyze_step(i + 1, step))
            lines.append("")
        
        return lines
    
    def _analyze_step(self, step_num: int, step: Step) -> List[str]:
        """Analyze a single step."""
        lines = [
            "",
            f"┌{'─' * 78}┐",
            f"│ STEP {step_num}: {step.name:<66} │",
            f"│ Type: {step.step_type or 'unknown':<68} │",
            f"└{'─' * 78}┘",
        ]
        
        # Inputs
        if step.inputs and step.inputs.data:
            lines.append("")
            lines.append("📥 INPUTS:")
            for key, value in step.inputs.data.items():
                value_str = self._format_value(value, max_length=60)
                lines.append(f"   • {key}: {value_str}")
            
            # Check for suspicious inputs
            if step.inputs.data.get("bad_match_mode"):
                self.issues_found.append(f"Step {step_num} ({step.name}): bad_match_mode is TRUE")
                lines.append(f"   ⚠️  WARNING: bad_match_mode is enabled!")
        
        # Candidates
        if step.candidates:
            lines.append("")
            lines.append(f"👥 CANDIDATES CONSIDERED: {step.candidates.total_count}")
            
            if step.candidates.items:
                top_candidates = sorted(
                    step.candidates.items, 
                    key=lambda c: c.score or 0, 
                    reverse=True
                )[:5]
                
                lines.append("   Top 5 by score:")
                for j, c in enumerate(top_candidates, 1):
                    score_str = f"{c.score:.3f}" if c.score else "N/A"
                    title = c.data.get("title", c.id)[:40] if c.data else c.id
                    lines.append(f"   {j}. [{score_str}] {title}")
                    
                    # Show explanation if available
                    if c.data and c.data.get("explanation"):
                        exp = c.data["explanation"][:60]
                        lines.append(f"      └─ {exp}")
        
        # Filters and Rejections
        if step.filters:
            lines.append("")
            lines.append("🔍 FILTERS APPLIED:")
            
            for f in step.filters:
                lines.append(f"   • {f.name or f.type}")
                
                if f.metrics:
                    removed = f.metrics.input_count - f.metrics.output_count
                    lines.append(
                        f"     Input: {f.metrics.input_count} → Output: {f.metrics.output_count} "
                        f"(removed {removed})"
                    )
                
                if f.config:
                    config_str = json.dumps(f.config, default=str)
                    if len(config_str) > 60:
                        config_str = config_str[:57] + "..."
                    lines.append(f"     Config: {config_str}")
                
                # Show rejection reasons (sample)
                if f.removed_reasons:
                    lines.append(f"     Rejection reasons ({len(f.removed_reasons)} total):")
                    sample_reasons = dict(list(f.removed_reasons.items())[:3])
                    for cid, reason in sample_reasons.items():
                        reason_short = reason[:50] if len(reason) > 50 else reason
                        lines.append(f"       - {cid}: {reason_short}")
                    
                    if len(f.removed_reasons) > 3:
                        lines.append(f"       ... and {len(f.removed_reasons) - 3} more")
        
        # Outputs
        if step.outputs:
            lines.append("")
            lines.append(f"📤 OUTPUTS: {step.outputs.count} results")
            
            if step.outputs.result_ids and len(step.outputs.result_ids) <= 10:
                lines.append(f"   Result IDs: {', '.join(step.outputs.result_ids[:5])}")
                if len(step.outputs.result_ids) > 5:
                    lines.append(f"   ... and {len(step.outputs.result_ids) - 5} more")
        
        # Reasoning
        if step.reasoning:
            lines.append("")
            lines.append("🧠 REASONING:")
            
            if step.reasoning.algorithm:
                lines.append(f"   Algorithm: {step.reasoning.algorithm}")
            
            if step.reasoning.confidence is not None:
                confidence = step.reasoning.confidence
                confidence_bar = self._generate_confidence_bar(confidence)
                lines.append(f"   Confidence: {confidence:.2f} {confidence_bar}")
                
                # Flag low confidence
                if confidence < 0.5:
                    self.issues_found.append(
                        f"Step {step_num} ({step.name}): LOW CONFIDENCE ({confidence:.2f})"
                    )
                    lines.append(f"   ⚠️  LOW CONFIDENCE - Results may be unreliable!")
            
            if step.reasoning.explanation:
                exp_lines = self._wrap_text(step.reasoning.explanation, 70)
                lines.append(f"   Explanation:")
                for exp_line in exp_lines:
                    lines.append(f"      {exp_line}")
                
                # Check for warning keywords
                if "[WARNING" in step.reasoning.explanation:
                    self.issues_found.append(
                        f"Step {step_num} ({step.name}): Warning in reasoning"
                    )
                if "[POTENTIAL BAD MATCH" in step.reasoning.explanation:
                    self.issues_found.append(
                        f"Step {step_num} ({step.name}): Potential bad match flagged"
                    )
            
            if step.reasoning.factors:
                lines.append("   Factors:")
                for factor in step.reasoning.factors:
                    weight = f" (weight: {factor.weight:.1f})" if factor.weight else ""
                    lines.append(f"      • {factor.name}: {factor.value:.2f}{weight}")
        
        return lines
    
    def _generate_issues_summary(self) -> List[str]:
        """Generate issues and recommendations summary."""
        lines = [
            "═" * 80,
            " ISSUES & RECOMMENDATIONS ".center(80, "═"),
            "═" * 80,
            "",
        ]
        
        if self.issues_found:
            lines.append("🔴 ISSUES FOUND:")
            for i, issue in enumerate(self.issues_found, 1):
                lines.append(f"   {i}. {issue}")
            lines.append("")
        else:
            lines.append("✅ No critical issues detected.")
            lines.append("")
        
        if self.warnings:
            lines.append("⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"   {i}. {warning}")
            lines.append("")
        
        # Recommendations based on issues
        if self.issues_found:
            lines.append("📋 RECOMMENDATIONS:")
            
            if any("bad_match_mode" in issue for issue in self.issues_found):
                lines.append("   1. Disable bad_match_mode in production")
                lines.append("      → This flag inverts scoring logic for testing purposes")
            
            if any("LOW CONFIDENCE" in issue for issue in self.issues_found):
                lines.append("   2. Investigate low confidence steps")
                lines.append("      → Review the LLM prompts and scoring model")
                lines.append("      → Consider adding more relevance factors")
            
            if any("bad match flagged" in issue.lower() for issue in self.issues_found):
                lines.append("   3. Check the final selection criteria")
                lines.append("      → The system detected a potential quality issue")
        
        lines.append("")
        lines.append("─" * 80)
        lines.append("End of Debug Report")
        lines.append("─" * 80)
        
        return lines
    
    # Helper methods
    
    def _get_status_icon(self, step: Step) -> str:
        """Get status icon for step."""
        icons = {
            "success": "✅",
            "failed": "❌",
            "running": "🔄",
            "pending": "⏳",
            "skipped": "⏭️",
        }
        return icons.get(step.status.value, "❓")
    
    def _format_value(self, value: Any, max_length: int = 50) -> str:
        """Format a value for display."""
        if isinstance(value, list):
            if len(value) <= 3:
                return str(value)
            return f"[{', '.join(str(v) for v in value[:3])}, ... ({len(value)} items)]"
        elif isinstance(value, dict):
            s = json.dumps(value, default=str)
            if len(s) > max_length:
                return s[:max_length - 3] + "..."
            return s
        else:
            s = str(value)
            if len(s) > max_length:
                return s[:max_length - 3] + "..."
            return s
    
    def _generate_confidence_bar(self, confidence: float) -> str:
        """Generate a visual confidence bar."""
        bar_length = 10
        filled = int(confidence * bar_length)
        empty = bar_length - filled
        
        if confidence >= 0.7:
            return f"[{'█' * filled}{'░' * empty}] 🟢"
        elif confidence >= 0.4:
            return f"[{'█' * filled}{'░' * empty}] 🟡"
        else:
            return f"[{'█' * filled}{'░' * empty}] 🔴"
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines


# =============================================================================
# Modified Competitor Selection for Capturing Run ID
# =============================================================================

def run_competitor_selection_with_run_id(bad_match_mode: bool = True) -> Optional[str]:
    """
    Run the competitor selection pipeline and return the run_id.
    
    This is a simplified version that tracks the run_id for debugging.
    """
    from examples.competitor_selection import (
        CompetitorSelectionPipeline, 
        InputProduct
    )
    
    # Create input product
    input_product = InputProduct(
        title="Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        category="Headphones",
        price=349.99,
    )
    
    print("Running competitor selection pipeline...")
    print(f"  Input: {input_product.title}")
    print(f"  Bad Match Mode: {bad_match_mode}")
    print()
    
    pipeline = CompetitorSelectionPipeline()
    
    # We need to access the run_id from the context manager
    # The easiest way is to manually control the run:
    run_tags = ["competitor-selection", "debug-demo"]
    if bad_match_mode:
        run_tags.append("bad-match-demo")
    
    client = pipeline.client
    
    # Start run manually to capture run_id
    run_builder = client.start_run(
        name="Competitor Selection Pipeline (Debug)",
        user_id="debug_user",
        session_id=f"debug_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=run_tags,
        blocking=True,  # Wait for API to get actual run_id
        context={
            "custom": {
                "input_product": input_product.title,
                "bad_match_mode": bad_match_mode,
            }
        }
    )
    
    run_id = run_builder.run_id
    print(f"📝 Run ID: {run_id}")
    
    try:
        # Run the pipeline steps (simplified version)
        # Step 1: Keywords
        keywords = pipeline.llm.generate_search_keywords(input_product)
        step1 = run_builder.add_step(
            name="Generate Search Keywords",
            step_type="llm_generation",
            inputs={"product_title": input_product.title}
        )
        step1.complete(count=len(keywords))
        print(f"  ✓ Step 1: Generated {len(keywords)} keywords")
        
        # Step 2: Search
        search_results = pipeline.search_api.search_products(
            keywords=keywords,
            category=input_product.category,
            limit=100
        )
        step2 = run_builder.add_step(
            name="Search Product Catalog",
            step_type="candidate_generation",
            inputs={"keywords": keywords, "limit": 100}
        )
        step2.add_candidates(
            [{"id": p.id, "score": p.rating / 5.0, "data": {"title": p.title}} 
             for p in search_results],
            source="catalog_api"
        )
        step2.complete(result_ids=[p.id for p in search_results], count=len(search_results))
        print(f"  ✓ Step 2: Found {len(search_results)} products")
        
        # Step 3: Filter
        min_price = input_product.price * 0.7
        max_price = input_product.price * 1.3
        min_rating = 3.0 if bad_match_mode else 3.5
        
        step3 = run_builder.add_step(
            name="Apply Price and Rating Filters",
            step_type="filtering",
            inputs={
                "input_count": len(search_results),
                "min_price": min_price,
                "max_price": max_price,
                "min_rating": min_rating,
            }
        )
        
        # Apply filters
        filtered = []
        price_rejected = {}
        rating_rejected = {}
        
        for p in search_results:
            if p.price < min_price:
                price_rejected[p.id] = f"Price ${p.price:.2f} below ${min_price:.2f}"
            elif p.price > max_price:
                price_rejected[p.id] = f"Price ${p.price:.2f} above ${max_price:.2f}"
            elif p.rating < min_rating:
                rating_rejected[p.id] = f"Rating {p.rating} below {min_rating}"
            else:
                filtered.append(p)
        
        step3.add_filter(
            filter_type="price_range",
            name="Price Range Filter",
            config={"min": min_price, "max": max_price},
            input_count=len(search_results),
            output_count=len(search_results) - len(price_rejected),
            removed_reasons=price_rejected
        )
        step3.add_filter(
            filter_type="min_rating",
            name="Minimum Rating Filter",
            config={"min_rating": min_rating},
            input_count=len(search_results) - len(price_rejected),
            output_count=len(filtered),
            removed_reasons=rating_rejected
        )
        
        for pid, reason in price_rejected.items():
            step3.reject_candidate(pid, reason, "price_range")
        for pid, reason in rating_rejected.items():
            step3.reject_candidate(pid, reason, "min_rating")
        
        step3.complete(result_ids=[p.id for p in filtered], count=len(filtered))
        print(f"  ✓ Step 3: Filtered to {len(filtered)} products")
        
        # Step 4: Rank
        step4 = run_builder.add_step(
            name="LLM Relevance Ranking",
            step_type="ranking",
            inputs={
                "input_count": len(filtered),
                "bad_match_mode": bad_match_mode,
            }
        )
        
        scored = []
        for p in filtered:
            score, explanation = pipeline.llm.score_relevance(
                input_product, p, bad_match_mode=bad_match_mode
            )
            p.relevance_score = score
            scored.append((p, explanation))
        
        scored.sort(key=lambda x: x[0].relevance_score or 0, reverse=True)
        
        step4.add_candidates(
            [{"id": p.id, "score": p.relevance_score, "data": {"title": p.title, "explanation": exp}}
             for p, exp in scored],
            source="llm_scoring"
        )
        
        avg_score = sum(p.relevance_score or 0 for p, _ in scored) / len(scored)
        step4.set_reasoning(
            algorithm="llm_relevance_ranking",
            explanation=(
                f"Scored {len(scored)} candidates. Avg score: {avg_score:.3f}"
                + (" [WARNING: bad_match_mode enabled!]" if bad_match_mode else "")
            ),
            confidence=0.3 if bad_match_mode else 0.85,
            factors=[
                {"name": "category_match", "value": 0.4, "weight": 0.4},
                {"name": "price_similarity", "value": 0.3, "weight": 0.3},
                {"name": "title_overlap", "value": 0.3, "weight": 0.3},
            ]
        )
        
        step4.complete(result_ids=[p.id for p, _ in scored], count=len(scored))
        print(f"  ✓ Step 4: Ranked {len(scored)} products")
        
        # Step 5: Select
        best = scored[0][0]
        best_exp = scored[0][1]
        
        step5 = run_builder.add_step(
            name="Select Best Competitor",
            step_type="selection",
            inputs={"input_count": len(scored)}
        )
        
        step5.add_candidates(
            [{"id": best.id, "score": best.relevance_score, 
              "data": {"title": best.title, "price": best.price, "rating": best.rating}}],
            source="ranked_candidates"
        )
        
        step5.set_reasoning(
            algorithm="best_score_selection",
            explanation=(
                f"Selected '{best.title}' with score {best.relevance_score:.3f}"
                + (" [POTENTIAL BAD MATCH: low confidence scoring]" if bad_match_mode else "")
            ),
            confidence=best.relevance_score or 0.5
        )
        
        step5.complete(result_ids=[best.id], count=1)
        print(f"  ✓ Step 5: Selected '{best.title}'")
        
        run_builder.complete(
            success=True,
            result_ids=[best.id],
            result_data={"selected_product": best.title}
        )
        
        print()
        print(f"🎯 Selected: {best.title}")
        print(f"   Score: {best.relevance_score:.3f}")
        print(f"   Price: ${best.price:.2f}")
        
    except Exception as e:
        run_builder.fail(error=str(e))
        print(f"❌ Pipeline failed: {e}")
        raise
    finally:
        # Give async requests time to complete
        time.sleep(1)
    
    return run_id


def fetch_and_debug_run(client: XRayClient, run_id: str) -> None:
    """Fetch run data from API and generate debug report."""
    print("\n" + "=" * 80)
    print("Fetching run data from X-Ray API...")
    print("=" * 80 + "\n")
    
    try:
        run = client.get_run(run_id, include_steps=True)
        
        # Generate debug report
        generator = DebugReportGenerator(run)
        report = generator.generate_report()
        
        print(report)
        
    except Exception as e:
        print(f"❌ Failed to fetch run from API: {e}")
        print()
        print("The X-Ray API server may not be running.")
        print("To start the API server, run:")
        print("  cd c:\\Company\\xray_system")
        print("  uvicorn api.main:app --reload")
        print()
        print("For demonstration, here's an example of what the debug report would show:")
        print()
        print_example_report()


def print_example_report():
    """Print an example debug report for demonstration."""
    example = """
════════════════════════════════════════════════════════════════════════════════
============================== X-RAY DEBUG REPORT ==============================
════════════════════════════════════════════════════════════════════════════════

Run ID:     run_abc123xyz
Run Name:   Competitor Selection Pipeline (Debug)
Status:     completed
Created:    2026-01-05 09:55:00
User ID:    debug_user
Session ID: debug_session_20260105_095500
⚠️  BAD MATCH MODE: ENABLED (intentional bad scoring)

────────────────────────────────────────────────────────────────────────────────
─────────────────────────────────── RUN SUMMARY ────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
Total Steps: 5

Decision Flow:
  ├── Step 1: Generate Search Keywords [llm_generation] → 5 outputs ✅
  ├── Step 2: Search Product Catalog [candidate_generation] → 100 outputs ✅
  ├── Step 3: Apply Price and Rating Filters [filtering] → 34 outputs ✅
  ├── Step 4: LLM Relevance Ranking [ranking] → 34 outputs ✅
  └── Step 5: Select Best Competitor [selection] → 1 outputs ✅

════════════════════════════════════════════════════════════════════════════════
============================ STEP-BY-STEP ANALYSIS =============================
════════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: LLM Relevance Ranking                                                │
│ Type: ranking                                                                │
└──────────────────────────────────────────────────────────────────────────────┘

📥 INPUTS:
   • input_count: 34
   • ranking_model: mock-relevance-v1
   • bad_match_mode: True
   ⚠️  WARNING: bad_match_mode is enabled!

👥 CANDIDATES CONSIDERED: 34
   Top 5 by score:
   1. [0.749] Apple Advanced Bluetooth Earbuds
      └─ Category: no match, Price diff: 29.7%, Word overlap: 0.0%...
   2. [0.712] Dell Home Smart Device
      └─ Category: no match, Price diff: 35.2%, Word overlap: 0.0%...
   3. [0.698] Samsung Premium Wireless Speaker
      └─ Category: no match, Price diff: 28.1%, Word overlap: 8.3%...

🧠 REASONING:
   Algorithm: llm_relevance_ranking
   Confidence: 0.30 [███░░░░░░░] 🔴
   ⚠️  LOW CONFIDENCE - Results may be unreliable!
   Explanation:
      Scored 34 candidates using LLM relevance model. Score range:
      0.309-0.749, avg: 0.521 [WARNING: bad_match_mode enabled!]
   Factors:
      • category_match: 0.40 (weight: 0.4)
      • price_similarity: 0.30 (weight: 0.3)
      • title_overlap: 0.30 (weight: 0.3)

════════════════════════════════════════════════════════════════════════════════
========================= ISSUES & RECOMMENDATIONS =============================
════════════════════════════════════════════════════════════════════════════════

🔴 ISSUES FOUND:
   1. Step 4 (LLM Relevance Ranking): bad_match_mode is TRUE
   2. Step 4 (LLM Relevance Ranking): LOW CONFIDENCE (0.30)
   3. Step 4 (LLM Relevance Ranking): Warning in reasoning
   4. Step 5 (Select Best Competitor): Potential bad match flagged

⚠️  WARNINGS:
   1. bad_match_mode is enabled - scoring is intentionally inverted

📋 RECOMMENDATIONS:
   1. Disable bad_match_mode in production
      → This flag inverts scoring logic for testing purposes
   2. Investigate low confidence steps
      → Review the LLM prompts and scoring model
      → Consider adding more relevance factors
   3. Check the final selection criteria
      → The system detected a potential quality issue

────────────────────────────────────────────────────────────────────────────────
End of Debug Report
────────────────────────────────────────────────────────────────────────────────
"""
    print(example)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the debug demonstration."""
    print("=" * 80)
    print(" X-RAY DEBUG BAD MATCH DEMONSTRATION ".center(80))
    print("=" * 80)
    print()
    print("This demo will:")
    print("  1. Run the competitor selection pipeline with bad_match_mode=True")
    print("  2. Capture the run_id")
    print("  3. Query the X-Ray API to fetch the run details")
    print("  4. Generate a human-readable debug report")
    print()
    print("─" * 80)
    print()
    
    # Run the pipeline
    run_id = run_competitor_selection_with_run_id(bad_match_mode=True)
    
    if run_id and run_id != "pending":
        # Create a client to fetch the run
        client = XRayClient(base_url="http://localhost:8000/api/v1")
        
        try:
            fetch_and_debug_run(client, run_id)
        finally:
            client.close()
    else:
        print("\n⚠️  Could not get run_id (API may not be running)")
        print("Showing example report instead...\n")
        print_example_report()


if __name__ == "__main__":
    main()
