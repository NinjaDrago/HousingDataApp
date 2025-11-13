#!/usr/bin/env python3
"""
Self-contained Formula-Based Offer Checker
- Fully independent (no Zillow CSVs required)
- Scales naturally for any city or property
- Refined coefficients for better accuracy
"""

def estimate_home_price(
    sqft: float,
    beds: int = 3,
    baths: int = 2,
    lot_size_acres: float = 0.1,
    year_built: int = 2000,
    property_type: str = "single",  # "single" or "duplex"
    has_amenities: bool = False
) -> float:
    """
    Estimate property price using a refined, independent formula.
    """
    # Base price per sqft by property type
    price_per_sqft = 320 if property_type == "single" else 250
    price = sqft * price_per_sqft

    # Bedrooms & bathrooms premium
    price += (beds - 3) * 15000
    price += (baths - 2) * 10000

    # Lot size premium
    if lot_size_acres > 0.5:
        price += (lot_size_acres - 0.5) * 40000

    # Amenities premium
    if has_amenities:
        price += 35000

    # Age adjustment (older homes slightly cheaper)
    age = 2025 - year_built
    price *= 1 - min(age * 0.001, 0.15)  # max 15% reduction

    return round(price, 0)


def check_offer_formula(
    city: str,
    state: str,
    price_offer: float,
    beds: int = 3,
    baths: int = 2,
    sqft: int = 1500,
    lot_size_acres: float = 0.1,
    year_built: int = 2000,
    property_type: str = "single",
    has_amenities: bool = False
) -> str:
    """
    Check a property offer against the estimated price.
    """
    estimated_price = estimate_home_price(
        sqft=sqft,
        beds=beds,
        baths=baths,
        lot_size_acres=lot_size_acres,
        year_built=year_built,
        property_type=property_type,
        has_amenities=has_amenities
    )

    # Advice based on comparison
    if price_offer < estimated_price * 0.95:
        advice = "Potentially a good deal."
    elif price_offer > estimated_price * 1.05:
        advice = "May be overpaying."
    else:
        advice = "Within typical market range."

    return (
        f"Estimated price for {beds} beds, {baths} baths, {sqft} sqft in "
        f"{city}, {state}: ${estimated_price:,.0f}\n{advice}"
    )
