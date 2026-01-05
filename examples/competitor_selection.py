"""
Competitor Selection Pipeline Example.

This example demonstrates a multi-step pipeline for finding competitor products
using the X-Ray SDK to track candidates at each step with detailed rejection reasons.

The pipeline:
    1. Generate search keywords from product info (mock LLM call)
    2. Search products (mock API returning 100 products)
    3. Filter by price range and minimum rating (reduces to ~30)
    4. Rank by relevance using LLM scoring (mock)
    5. Select the best matching competitor

Includes a "BAD match" scenario to demonstrate X-Ray debugging capabilities.
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from xray_sdk import (
    XRayClient,
    Candidate,
    StepFilter,
    ReasoningFactor,
)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Product:
    """Represents a product in our catalog."""
    id: str
    title: str
    category: str
    price: float
    rating: float
    review_count: int
    brand: str
    features: List[str]
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "price": self.price,
            "rating": self.rating,
            "review_count": self.review_count,
            "brand": self.brand,
            "features": self.features,
            "relevance_score": self.relevance_score,
        }


@dataclass
class InputProduct:
    """The product we want to find competitors for."""
    title: str
    category: str
    price: float


# =============================================================================
# Mock Services
# =============================================================================

class MockLLMService:
    """Simulates an LLM API for keyword generation and relevance scoring."""
    
    @staticmethod
    def generate_search_keywords(product: InputProduct) -> List[str]:
        """
        Mock LLM call to generate search keywords from product info.
        In production, this would call GPT-4, Claude, etc.
        """
        time.sleep(0.1)  # Simulate API latency
        
        # Extract keywords from title
        title_words = product.title.lower().split()
        keywords = [w for w in title_words if len(w) > 3]
        
        # Add category-based keywords
        category_keywords = {
            "electronics": ["tech", "gadget", "device", "electronic"],
            "headphones": ["audio", "sound", "wireless", "bluetooth", "headset"],
            "laptops": ["computer", "notebook", "portable", "pc"],
            "smartphones": ["phone", "mobile", "cellular", "device"],
            "cameras": ["photography", "digital", "lens", "imaging"],
        }
        
        cat_lower = product.category.lower()
        for cat_key, cat_words in category_keywords.items():
            if cat_key in cat_lower:
                keywords.extend(cat_words[:2])
                break
        
        return list(set(keywords))[:5]  # Limit to 5 unique keywords
    
    @staticmethod
    def score_relevance(
        query_product: InputProduct, 
        candidate: Product,
        bad_match_mode: bool = False
    ) -> Tuple[float, str]:
        """
        Mock LLM call to score relevance between products.
        Returns (score, explanation).
        
        In bad_match_mode, intentionally returns high scores for poor matches
        to demonstrate debugging scenarios.
        """
        time.sleep(0.01)  # Simulate API latency
        
        # Base relevance factors
        category_match = query_product.category.lower() in candidate.category.lower()
        price_diff_pct = abs(query_product.price - candidate.price) / max(query_product.price, 1)
        
        # Title similarity (simple word overlap)
        query_words = set(query_product.title.lower().split())
        candidate_words = set(candidate.title.lower().split())
        word_overlap = len(query_words & candidate_words) / max(len(query_words | candidate_words), 1)
        
        if bad_match_mode:
            # In bad match mode, invert the scoring logic to produce BAD results
            # This simulates a bug in the LLM scoring prompt or model issues
            score = (
                (0.9 if not category_match else 0.3) * 0.4 +  # Prefer wrong category
                (1.0 - word_overlap) * 0.3 +                   # Prefer no word overlap
                price_diff_pct * 0.3                           # Prefer high price diff
            )
            explanation = (
                f"Category: {'match' if category_match else 'no match'}, "
                f"Price diff: {price_diff_pct:.1%}, "
                f"Word overlap: {word_overlap:.1%} "
                f"[BAD_MATCH_MODE: scoring inverted]"
            )
        else:
            # Normal scoring
            score = (
                (0.5 if category_match else 0.0) * 0.4 +
                (1.0 - min(price_diff_pct, 1.0)) * 0.3 +
                word_overlap * 0.3
            )
            explanation = (
                f"Category: {'match' if category_match else 'no match'}, "
                f"Price diff: {price_diff_pct:.1%}, "
                f"Word overlap: {word_overlap:.1%}"
            )
        
        return round(score, 3), explanation


class MockProductSearchAPI:
    """Simulates a product search API (e.g., internal catalog, competitor API)."""
    
    # Sample product data for generating mock results
    BRANDS = [
        "Sony", "Samsung", "Apple", "Bose", "JBL", "LG", "Anker", 
        "Sennheiser", "Dell", "HP", "Lenovo", "Microsoft", "Google"
    ]
    
    ADJECTIVES = [
        "Pro", "Ultra", "Max", "Elite", "Premium", "Advanced", 
        "Classic", "Essential", "Studio", "Home", "Portable"
    ]
    
    CATEGORY_PRODUCTS = {
        "headphones": [
            "Wireless Headphones", "Bluetooth Earbuds", "Noise Cancelling Headphones",
            "Gaming Headset", "Sports Earphones", "Studio Headphones"
        ],
        "electronics": [
            "Smart Device", "Wireless Speaker", "Portable Charger",
            "Smart Display", "Wireless Controller", "USB Hub"
        ],
        "laptops": [
            "Business Laptop", "Gaming Laptop", "Ultrabook",
            "Chromebook", "Workstation", "2-in-1 Laptop"
        ],
    }
    
    FEATURES_POOL = [
        "Wireless", "Bluetooth 5.0", "Fast Charging", "Water Resistant",
        "Noise Cancelling", "Long Battery", "Ergonomic", "Lightweight",
        "Premium Build", "Touch Controls", "Voice Assistant", "USB-C"
    ]
    
    @classmethod
    def search_products(
        cls, 
        keywords: List[str], 
        category: str, 
        limit: int = 100
    ) -> List[Product]:
        """
        Mock product search API that returns ~100 products.
        In production, this would call an actual search service.
        """
        time.sleep(0.2)  # Simulate API latency
        
        products = []
        cat_key = next(
            (k for k in cls.CATEGORY_PRODUCTS if k in category.lower()), 
            "electronics"
        )
        product_types = cls.CATEGORY_PRODUCTS.get(cat_key, cls.CATEGORY_PRODUCTS["electronics"])
        
        for i in range(limit):
            brand = random.choice(cls.BRANDS)
            adj = random.choice(cls.ADJECTIVES)
            product_type = random.choice(product_types)
            
            # Add keyword-based variation
            keyword_bonus = random.choice(keywords) if keywords and random.random() > 0.5 else ""
            title = f"{brand} {adj} {product_type}"
            if keyword_bonus:
                title += f" {keyword_bonus.title()}"
            
            # Randomize but keep realistic distributions
            price = round(random.uniform(20, 500) * (1 + random.gauss(0, 0.3)), 2)
            rating = round(min(5.0, max(1.0, random.gauss(4.0, 0.7))), 1)
            review_count = int(random.expovariate(0.001)) + random.randint(5, 100)
            
            features = random.sample(cls.FEATURES_POOL, k=random.randint(2, 5))
            
            product = Product(
                id=f"prod_{i:04d}",
                title=title,
                category=cat_key.title() if random.random() > 0.1 else "Other",  # 10% wrong category
                price=price,
                rating=rating,
                review_count=review_count,
                brand=brand,
                features=features,
            )
            products.append(product)
        
        return products


# =============================================================================
# Main Pipeline
# =============================================================================

class CompetitorSelectionPipeline:
    """
    Multi-step pipeline for finding competitor products.
    Uses X-Ray SDK to track candidates and decisions at each step.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.client = XRayClient(base_url=base_url)
        self.llm = MockLLMService()
        self.search_api = MockProductSearchAPI()
    
    def find_competitors(
        self, 
        product: InputProduct,
        price_tolerance: float = 0.3,  # 30% price range
        min_rating: float = 3.5,
        bad_match_mode: bool = False,  # Enable to demonstrate debugging
    ) -> Optional[Product]:
        """
        Find the best competitor product for the given input product.
        
        Args:
            product: The product to find competitors for.
            price_tolerance: Percentage tolerance for price filtering.
            min_rating: Minimum rating threshold.
            bad_match_mode: If True, intentionally produces bad results for debugging demo.
            
        Returns:
            The best matching competitor product, or None if no suitable match found.
        """
        run_tags = ["competitor-selection", "demo"]
        if bad_match_mode:
            run_tags.append("bad-match-demo")
        
        with self.client.run(
            name="Competitor Selection Pipeline",
            user_id="demo_user",
            session_id=f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=run_tags,
            context={
                "custom": {
                    "input_product": {
                        "title": product.title,
                        "category": product.category,
                        "price": product.price,
                    },
                    "price_tolerance": price_tolerance,
                    "min_rating": min_rating,
                    "bad_match_mode": bad_match_mode,
                }
            }
        ) as run:
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Generate Search Keywords (Mock LLM)
            # ─────────────────────────────────────────────────────────────────
            step1 = run.add_step(
                name="Generate Search Keywords",
                step_type="llm_generation",
                inputs={
                    "product_title": product.title,
                    "product_category": product.category,
                    "product_price": product.price,
                    "prompt_template": "Extract search keywords for competitor analysis",
                }
            )
            
            keywords = self.llm.generate_search_keywords(product)
            
            step1.set_reasoning(
                algorithm="keyword_extraction",
                explanation=f"Extracted {len(keywords)} keywords from product info using LLM",
                confidence=0.85,
                factors=[
                    ReasoningFactor(name="title_parsing", value=0.9, weight=0.5),
                    ReasoningFactor(name="category_mapping", value=0.8, weight=0.5),
                ]
            )
            
            step1.complete(
                result_ids=None,
                count=len(keywords),
            )
            
            print(f"✓ Step 1: Generated {len(keywords)} keywords: {keywords}")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Search Products (Mock API - 100 products)
            # ─────────────────────────────────────────────────────────────────
            step2 = run.add_step(
                name="Search Product Catalog",
                step_type="candidate_generation",
                inputs={
                    "keywords": keywords,
                    "category": product.category,
                    "search_limit": 100,
                    "api_endpoint": "mock://product-search/v1",
                }
            )
            
            search_results = self.search_api.search_products(
                keywords=keywords,
                category=product.category,
                limit=100
            )
            
            # Track all candidates with summary mode (100 is a lot)
            step2.add_candidates(
                [
                    {
                        "id": p.id,
                        "score": p.rating / 5.0,  # Normalize rating as initial score
                        "data": p.to_dict(),
                    }
                    for p in search_results
                ],
                source="product_catalog_api",
                mode="detailed",  # Keep detailed since 100 is manageable
            )
            
            step2.complete(
                result_ids=[p.id for p in search_results],
                count=len(search_results),
            )
            
            print(f"✓ Step 2: Found {len(search_results)} products from catalog search")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Filter by Price and Rating
            # ─────────────────────────────────────────────────────────────────
            step3 = run.add_step(
                name="Apply Price and Rating Filters",
                step_type="filtering",
                inputs={
                    "input_count": len(search_results),
                    "price_tolerance": price_tolerance,
                    "min_rating": min_rating,
                    "reference_price": product.price,
                }
            )
            
            # Track initial candidates
            step3.add_candidates(
                [{"id": p.id, "data": {"title": p.title, "price": p.price, "rating": p.rating}} 
                 for p in search_results],
                source="previous_step"
            )
            
            # Price filter
            min_price = product.price * (1 - price_tolerance)
            max_price = product.price * (1 + price_tolerance)
            
            price_filtered = []
            price_rejected = []
            price_rejected_reasons = {}
            
            for p in search_results:
                if min_price <= p.price <= max_price:
                    price_filtered.append(p)
                else:
                    price_rejected.append(p)
                    if p.price < min_price:
                        reason = f"Price ${p.price:.2f} below minimum ${min_price:.2f}"
                    else:
                        reason = f"Price ${p.price:.2f} above maximum ${max_price:.2f}"
                    price_rejected_reasons[p.id] = reason
            
            step3.add_filter(
                filter_type="price_range",
                name="Price Range Filter",
                config={
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "reference_price": product.price,
                    "tolerance": price_tolerance,
                },
                input_count=len(search_results),
                output_count=len(price_filtered),
                removed_ids=[p.id for p in price_rejected],
                removed_reasons=price_rejected_reasons,
            )
            
            # Track rejections
            for p in price_rejected:
                step3.reject_candidate(
                    p.id,
                    reason=price_rejected_reasons[p.id],
                    filter_name="price_range"
                )
            
            # Rating filter
            rating_filtered = []
            rating_rejected = []
            rating_rejected_reasons = {}
            
            for p in price_filtered:
                if p.rating >= min_rating:
                    rating_filtered.append(p)
                else:
                    rating_rejected.append(p)
                    rating_rejected_reasons[p.id] = f"Rating {p.rating} below minimum {min_rating}"
            
            step3.add_filter(
                filter_type="min_rating",
                name="Minimum Rating Filter",
                config={
                    "min_rating": min_rating,
                },
                input_count=len(price_filtered),
                output_count=len(rating_filtered),
                removed_ids=[p.id for p in rating_rejected],
                removed_reasons=rating_rejected_reasons,
            )
            
            # Track rejections
            for p in rating_rejected:
                step3.reject_candidate(
                    p.id,
                    reason=rating_rejected_reasons[p.id],
                    filter_name="min_rating"
                )
            
            step3.complete(
                result_ids=[p.id for p in rating_filtered],
                count=len(rating_filtered),
            )
            
            print(f"✓ Step 3: Filtered to {len(rating_filtered)} products "
                  f"(rejected {len(price_rejected)} by price, {len(rating_rejected)} by rating)")
            
            if not rating_filtered:
                print("✗ No products passed filters!")
                return None
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Rank by Relevance (Mock LLM Scoring)
            # ─────────────────────────────────────────────────────────────────
            step4 = run.add_step(
                name="LLM Relevance Ranking",
                step_type="ranking",
                inputs={
                    "input_count": len(rating_filtered),
                    "ranking_model": "mock-relevance-v1",
                    "bad_match_mode": bad_match_mode,  # Track this for debugging!
                }
            )
            
            # Score each candidate
            scored_products = []
            score_explanations = {}
            
            for p in rating_filtered:
                score, explanation = self.llm.score_relevance(
                    product, p, bad_match_mode=bad_match_mode
                )
                p.relevance_score = score
                scored_products.append(p)
                score_explanations[p.id] = explanation
            
            # Sort by relevance score (descending)
            scored_products.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            
            # Track all candidates with their scores
            step4.add_candidates(
                [
                    {
                        "id": p.id,
                        "score": p.relevance_score,
                        "data": {
                            "title": p.title,
                            "price": p.price,
                            "rating": p.rating,
                            "explanation": score_explanations[p.id],
                        },
                    }
                    for p in scored_products
                ],
                source="llm_scoring"
            )
            
            # Calculate ranking factors
            avg_score = sum(p.relevance_score or 0 for p in scored_products) / len(scored_products)
            max_score = max(p.relevance_score or 0 for p in scored_products)
            min_score = min(p.relevance_score or 0 for p in scored_products)
            
            step4.set_reasoning(
                algorithm="llm_relevance_ranking",
                explanation=(
                    f"Scored {len(scored_products)} candidates using LLM relevance model. "
                    f"Score range: {min_score:.3f}-{max_score:.3f}, avg: {avg_score:.3f}"
                    + (" [WARNING: bad_match_mode enabled!]" if bad_match_mode else "")
                ),
                confidence=0.3 if bad_match_mode else 0.85,  # Low confidence in bad mode
                factors=[
                    ReasoningFactor(name="category_match", value=0.4, weight=0.4),
                    ReasoningFactor(name="price_similarity", value=0.3, weight=0.3),
                    ReasoningFactor(name="title_overlap", value=0.3, weight=0.3),
                ]
            )
            
            step4.complete(
                result_ids=[p.id for p in scored_products],
                count=len(scored_products),
            )
            
            print(f"✓ Step 4: Ranked {len(scored_products)} products by relevance "
                  f"(scores: {min_score:.3f} - {max_score:.3f})")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Select Best Match
            # ─────────────────────────────────────────────────────────────────
            step5 = run.add_step(
                name="Select Best Competitor",
                step_type="selection",
                inputs={
                    "input_count": len(scored_products),
                    "selection_strategy": "highest_relevance_score",
                }
            )
            
            # Track top candidates considered
            top_candidates = scored_products[:5]  # Top 5 for selection consideration
            
            step5.add_candidates(
                [
                    {
                        "id": p.id,
                        "score": p.relevance_score,
                        "data": {
                            "title": p.title,
                            "category": p.category,
                            "price": p.price,
                            "rating": p.rating,
                            "brand": p.brand,
                            "rank": i + 1,
                            "explanation": score_explanations[p.id],
                        },
                    }
                    for i, p in enumerate(top_candidates)
                ],
                source="ranked_candidates"
            )
            
            # Select the best match
            best_match = scored_products[0]
            
            # Track why we rejected alternatives (top 5)
            for i, p in enumerate(top_candidates[1:], start=2):
                step5.reject_candidate(
                    p.id,
                    reason=f"Ranked #{i} with score {p.relevance_score:.3f} vs best score {best_match.relevance_score:.3f}",
                    filter_name="rank_selection"
                )
            
            step5.set_reasoning(
                algorithm="best_score_selection",
                explanation=(
                    f"Selected '{best_match.title}' as best competitor match with "
                    f"score {best_match.relevance_score:.3f}. "
                    f"Price: ${best_match.price:.2f} vs input ${product.price:.2f}"
                    + (" [POTENTIAL BAD MATCH: low confidence scoring]" if bad_match_mode else "")
                ),
                confidence=best_match.relevance_score or 0.5,
                factors=[
                    ReasoningFactor(
                        name="relevance_score", 
                        value=best_match.relevance_score or 0, 
                        weight=0.6
                    ),
                    ReasoningFactor(
                        name="rating", 
                        value=best_match.rating / 5.0, 
                        weight=0.2
                    ),
                    ReasoningFactor(
                        name="review_count", 
                        value=min(best_match.review_count / 1000, 1.0), 
                        weight=0.2
                    ),
                ]
            )
            
            step5.complete(
                result_ids=[best_match.id],
                count=1,
            )
            
            print(f"✓ Step 5: Selected best match: '{best_match.title}'")
            print(f"          Price: ${best_match.price:.2f}, Rating: {best_match.rating}")
            print(f"          Relevance Score: {best_match.relevance_score:.3f}")
            
            return best_match
    
    def close(self):
        """Close the X-Ray client."""
        self.client.close()


# =============================================================================
# Demo Execution
# =============================================================================

def run_demo():
    """Run the competitor selection demo with both normal and BAD match modes."""
    print("=" * 70)
    print("COMPETITOR SELECTION PIPELINE DEMO")
    print("Using X-Ray SDK for Decision Tracking")
    print("=" * 70)
    
    # Create input product
    input_product = InputProduct(
        title="Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        category="Headphones",
        price=349.99,
    )
    
    print(f"\nInput Product:")
    print(f"  Title: {input_product.title}")
    print(f"  Category: {input_product.category}")
    print(f"  Price: ${input_product.price}")
    
    pipeline = CompetitorSelectionPipeline()
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # RUN 1: Normal Mode (should produce good match)
        # ─────────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("RUN 1: Normal Mode (Good Match Expected)")
        print("─" * 70 + "\n")
        
        result = pipeline.find_competitors(
            input_product,
            price_tolerance=0.3,
            min_rating=3.5,
            bad_match_mode=False,
        )
        
        if result:
            print(f"\n🎯 RESULT: {result.title}")
            print(f"   Price: ${result.price:.2f} (vs ${input_product.price:.2f})")
            print(f"   Rating: {result.rating}/5.0")
            print(f"   Relevance: {result.relevance_score:.3f}")
        else:
            print("\n❌ No suitable competitor found.")
        
        # ─────────────────────────────────────────────────────────────────────
        # RUN 2: Bad Match Mode (demonstrates debugging)
        # ─────────────────────────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("RUN 2: Bad Match Mode (Debugging Demonstration)")
        print("─" * 70 + "\n")
        print("⚠️  This run intentionally uses BAD scoring logic to demonstrate")
        print("   how X-Ray helps debug poor results by tracking all decisions.\n")
        
        bad_result = pipeline.find_competitors(
            input_product,
            price_tolerance=0.3,
            min_rating=3.0,  # Lower rating to get more candidates
            bad_match_mode=True,  # Enable bad scoring!
        )
        
        if bad_result:
            print(f"\n🔴 BAD RESULT: {bad_result.title}")
            print(f"   Price: ${bad_result.price:.2f} (vs ${input_product.price:.2f})")
            print(f"   Rating: {bad_result.rating}/5.0")
            print(f"   Relevance: {bad_result.relevance_score:.3f}")
            print("\n   ℹ️  This is likely a poor match! Check X-Ray dashboard to see:")
            print("   • Step 4 inputs show 'bad_match_mode: true'")
            print("   • Step 4 reasoning shows low confidence (0.3)")
            print("   • Step 4 explanation includes '[WARNING: bad_match_mode enabled!]'")
            print("   • Step 5 explanation includes '[POTENTIAL BAD MATCH]'")
        
    finally:
        pipeline.close()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - Check X-Ray API for detailed run traces!")
    print("View runs at: http://localhost:8000/api/v1/runs")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
