"""
synthetic_data.py

Provides synthetic documents about Athene's actual business — annuities and
retirement services — to replace PyPDFLoader in practice mode.

Documents cover:
  - Fixed Indexed Annuities (FIA)
  - Multi-Year Guaranteed Annuities (MYGA)
  - Guaranteed Lifetime Withdrawal Benefit (GLWB) riders
  - Pension Risk Transfer (PRT)
  - Surrender charges and free-withdrawal provisions
  - Tax treatment and RMD rules
  - Suitability and compliance guidelines
  - Athene product comparison / agent guide

In the real test, these would be loaded via:
    loader = PyPDFLoader("path/to/product_guide.pdf")
    docs = loader.load()
"""

from langchain_core.documents import Document


def get_synthetic_documents() -> list[Document]:
    """
    Return a list of LangChain Document objects representing synthetic
    Athene annuity and retirement product documentation.
    Each document carries 'source', 'page', and 'doc_type' metadata —
    identical to what PyPDFLoader would produce.

    Returns:
        List of Document objects ready for chunking and indexing.
    """
    raw_docs = [

        # ── Fixed Indexed Annuity (FIA) Overview ─────────────────────────
        (
            """Fixed Indexed Annuity (FIA) — Product Overview

A Fixed Indexed Annuity (FIA) is a contract issued by an insurance company
that offers principal protection combined with the potential for interest
crediting linked to the performance of a market index (e.g., S&P 500,
Russell 2000, MSCI EAFE). The contract owner does not directly participate
in the market; instead, interest is credited based on a formula tied to
index performance, subject to a cap, participation rate, or spread.

How Interest Is Credited:
- Point-to-Point (Annual): Interest is calculated on the difference between
  the index value at the start and end of the crediting period (typically
  one year). If the index gains 12% and the cap is 8%, the credited rate
  is 8%. If the index declines, 0% is credited — principal is protected.
- Monthly Sum: Monthly index changes are summed over 12 months, subject to
  a monthly cap (e.g., 2%). Gains and losses are both included in the sum.
- Participation Rate: Owner receives a defined percentage of index gains
  (e.g., 60% participation, no cap). If index gains 10%, credited rate = 6%.

Principal Protection: The floor is 0% — the contract owner cannot lose
principal or previously credited interest due to market declines. This is
the primary distinguishing feature of an FIA versus a variable annuity.

Typical Surrender Period: 7–10 years. Surrender charges begin at 8–10%
in year 1 and decline to 0% at the end of the surrender period.

Use Case: Accumulation-phase clients aged 50–70 seeking market-linked
upside potential without downside risk.""",
            {"source": "fia_product_overview.pdf", "page": 1, "doc_type": "product"},
        ),

        # ── MYGA Overview ─────────────────────────────────────────────────
        (
            """Multi-Year Guaranteed Annuity (MYGA) — Product Overview

A Multi-Year Guaranteed Annuity (MYGA) is a fixed deferred annuity that
credits a guaranteed interest rate for a specified term, typically 2–10
years. It functions similarly to a bank CD but with tax-deferred growth
and is issued by an insurance company rather than a bank.

Key Features:
- Guaranteed Rate: The declared interest rate is locked for the entire
  surrender period. For example, an Athene MYGA with a 5-year term might
  offer a 5.40% guaranteed annual rate as of Q2 2025.
- Tax Deferral: Interest compounds tax-deferred until withdrawal. Unlike a
  CD, there is no annual 1099 unless a withdrawal is taken.
- Free Withdrawal Provision: After the first contract year, owners may
  withdraw up to 10% of the accumulated account value per year without
  incurring a surrender charge. Some contracts also waive surrender charges
  for terminal illness or nursing home confinement (90-day waiting period).
- Surrender Charges: Withdrawals exceeding the free-withdrawal amount
  during the surrender period are subject to charges. Example 5-year
  schedule: 8%, 7%, 6%, 5%, 4%, 0% (charges in years 1–5, then 0%).

Surrender at Maturity: At the end of the surrender period, the owner
receives a 30-day window to surrender, renew at the new declared rate,
or exchange into another product (1035 exchange) without penalty.

MYGA vs. CD: MYGAs offer tax deferral and generally higher rates than
bank CDs of equivalent term, but are not FDIC-insured. They are backed
by the claims-paying ability of the issuing insurance company and covered
by state guaranty associations (limits vary by state, typically $250K).

Use Case: Conservative clients seeking a predictable, tax-deferred
alternative to CDs or Treasury bonds during the accumulation phase.""",
            {"source": "myga_product_overview.pdf", "page": 1, "doc_type": "product"},
        ),

        # ── GLWB Rider ────────────────────────────────────────────────────
        (
            """Guaranteed Lifetime Withdrawal Benefit (GLWB) Rider

The Guaranteed Lifetime Withdrawal Benefit (GLWB) is an optional rider
available on Athene FIA products that guarantees the contract owner can
withdraw a specified percentage of the Benefit Base each year for life,
regardless of actual account value performance.

Benefit Base vs. Account Value:
- Account Value: The actual accumulated value of the contract (used for
  surrender, death benefit, and transfers). Can fluctuate based on
  credited interest and withdrawals.
- Benefit Base: A separate, notional value used only to calculate the
  guaranteed withdrawal amount. The Benefit Base is NOT available as a
  lump sum; it is purely a calculation tool.

Benefit Base Roll-Up: During the deferral (accumulation) phase, the
Benefit Base typically increases by a guaranteed roll-up rate (e.g., 7%
simple interest per year, or 6% compound) for each year withdrawals are
not taken, up to a maximum of 10–20 deferral years.

Guaranteed Withdrawal Percentage (GWP): Once the owner begins
withdrawals, the annual guaranteed withdrawal amount is:
  Annual Withdrawal = Benefit Base × GWP
  GWP is based on the owner's age at first withdrawal:
    Age 60–64: 4.5% per year
    Age 65–69: 5.0% per year
    Age 70–74: 5.5% per year
    Age 75+:   6.0% per year

Lifetime Guarantee: If the account value is depleted to $0 due to
ongoing guaranteed withdrawals (not excess withdrawals), Athene continues
paying the guaranteed withdrawal amount for the life of the annuitant.

Rider Cost: Typically 0.95%–1.10% of Benefit Base per year, deducted
from the contract's account value annually.

Excess Withdrawals: Withdrawals exceeding the annual guaranteed amount
reduce the Benefit Base proportionally and may permanently reduce future
guaranteed payments.""",
            {"source": "glwb_rider_guide.pdf", "page": 1, "doc_type": "product"},
        ),

        # ── Tax Treatment and RMDs ────────────────────────────────────────
        (
            """Annuity Tax Treatment and Required Minimum Distributions (RMDs)

Non-Qualified Annuities (funded with after-tax dollars):
- Growth accumulates tax-deferred until withdrawal.
- Withdrawals are taxed on a LIFO (Last In, First Out) basis: earnings
  come out first and are taxed as ordinary income; principal is returned
  tax-free after all gains are distributed.
- 10% IRS early withdrawal penalty applies to earnings withdrawn before
  age 59½, with exceptions for death, disability, and annuitization.
- Non-qualified annuities are NOT subject to RMDs during the owner's
  lifetime (unlike IRAs and qualified plans).
- 1035 Exchange: A non-qualified annuity can be exchanged tax-free for
  another annuity or a life insurance policy under IRS Section 1035.

Qualified Annuities (funded with pre-tax dollars, held in IRA or
employer plan):
- Contributions may be tax-deductible depending on plan type.
- All withdrawals are taxed as ordinary income.
- Subject to Required Minimum Distributions (RMDs) beginning at age 73
  (SECURE 2.0 Act, effective 2023; age 75 beginning in 2033).
- RMD Calculation: Prior year-end account value ÷ IRS Uniform Lifetime
  Table life expectancy factor.
- RMDs must be satisfied from the contract's account value each year;
  the GLWB guaranteed withdrawal amount may or may not satisfy the full
  RMD depending on the contract value and age.

Death Benefit and Step-Up:
- Upon the owner's death, the beneficiary receives the greater of the
  account value or the death benefit base (varies by contract).
- Spousal beneficiaries may continue the contract as owner.
- Non-spousal beneficiaries must take distributions within 10 years
  under the SECURE Act (2019).""",
            {"source": "tax_and_rmd_guide.pdf", "page": 1, "doc_type": "guide"},
        ),

        # ── Pension Risk Transfer ─────────────────────────────────────────
        (
            """Pension Risk Transfer (PRT) — Institutional Solutions

Pension Risk Transfer (PRT) is the process by which a defined benefit
(DB) pension plan sponsor (typically a corporation) transfers some or all
of its pension obligations to an insurance company through the purchase
of a group annuity contract. Athene is one of the largest PRT providers
in the United States.

Types of PRT Transactions:
- Buy-Out (Full Termination): The pension plan is fully terminated and all
  obligations transferred to the insurer. Participants become annuitants
  under the group annuity contract. The plan sponsor is fully discharged
  from pension liability.
- Buy-In: The plan sponsor purchases a group annuity as an asset of the
  plan. The insurer pays the plan, and the plan continues to pay
  participants. Risk is transferred but the plan is not terminated.
- Lift-Out (Partial Buy-Out): A subset of the pension liability (e.g.,
  retirees only) is transferred to the insurer. The remaining liability
  stays with the plan.

Key Drivers for Plan Sponsors:
- Balance sheet de-risking: Removes pension liability and associated
  interest rate and longevity risk from the corporate balance sheet.
- PBGC premium reduction: Pension Benefit Guaranty Corporation (PBGC)
  premiums are eliminated for transferred obligations.
- Funded status volatility: Eliminates mark-to-market earnings impact
  from pension asset/liability fluctuations.

Selection Criteria (Fiduciary Safe Harbor):
Per DOL Interpretive Bulletin 95-1, plan fiduciaries must select the
safest available annuity provider, considering: financial strength
ratings, size and diversification of assets, level of surplus, and the
insurer's experience and expertise in group annuities.

Athene's PRT Credentials: A-rated by A.M. Best; $200B+ AUM; completed
$10B+ in PRT transactions including notable deals with Bristol-Myers
Squibb, Lockheed Martin, and NCR.""",
            {"source": "prt_solutions_guide.pdf", "page": 1, "doc_type": "product"},
        ),

        # ── Suitability & Compliance ──────────────────────────────────────
        (
            """Annuity Suitability and Best Interest Compliance Guidelines

Regulatory Framework:
- SEC Regulation Best Interest (Reg BI): Effective June 2020. Requires
  broker-dealers to act in the client's best interest when recommending
  annuities, including documenting the basis for the recommendation.
- NAIC Model 275 (Suitability in Annuity Transactions): Adopted in most
  states. Requires agents to collect a client profile and document that
  the recommended product is suitable given the client's financial
  situation, needs, and objectives.
- New York Regulation 187: Stricter best-interest standard applying to
  annuity and life insurance sales in New York. Agents must document why
  the recommended product is in the client's best interest vs. alternatives.

Required Client Information (Suitability Profile):
1. Age and health status
2. Annual income and sources (employment, Social Security, pension)
3. Liquid assets and total net worth
4. Tax status and bracket
5. Financial objectives (accumulation, income, legacy)
6. Risk tolerance
7. Time horizon and liquidity needs
8. Existing annuity or life insurance holdings

Key Suitability Red Flags:
- Client over age 80 purchasing a product with a 10-year surrender period
- Annuity funded from a client's only liquid emergency fund
- Replacement of an existing annuity without documented benefit to client
- Client in high tax bracket purchasing a non-qualified annuity in an IRA
  (already tax-deferred — no additional tax benefit)

Documentation Requirements:
All recommendations must be supported by a completed Suitability
Questionnaire, signed by both the agent and the client, retained for a
minimum of 6 years after the transaction date.""",
            {"source": "suitability_compliance_guide.pdf", "page": 1, "doc_type": "guide"},
        ),

        # ── Surrender Charges and Liquidity ──────────────────────────────
        (
            """Surrender Charges, Liquidity Provisions, and Contract Exits

Surrender Charges:
Surrender charges (also called contingent deferred sales charges, or
CDSC) protect the insurance company's ability to invest in longer-duration
assets by discouraging early contract termination. They are expressed as
a percentage of the surrendered amount and decline over the surrender period.

Example 7-Year Surrender Schedule (Athene Agility 7):
  Year 1: 8.50%  |  Year 2: 8.00%  |  Year 3: 7.00%  |  Year 4: 6.00%
  Year 5: 5.00%  |  Year 6: 4.00%  |  Year 7: 3.00%  |  Year 8+: 0.00%

Free-Withdrawal Amount:
- Most Athene contracts allow penalty-free withdrawal of up to 10% of the
  account value (or premium, depending on contract) per contract year,
  beginning in contract year 2 (year 1 in some products).
- Free-withdrawal amounts not used in a given year do NOT accumulate; they
  reset annually.
- Free withdrawals taken from a GLWB contract count against the annual
  guaranteed withdrawal amount for that year.

Surrender Charge Waivers (Common Provisions):
- Terminal illness: Full surrender without charge if annuitant is
  diagnosed with a terminal illness (life expectancy ≤ 12 months).
- Nursing home confinement: Waiver after 90 consecutive days of
  confinement in a licensed nursing facility.
- Death of owner: No surrender charge applied to death benefit proceeds.
- Some contracts also waive charges upon annuitization.

Market Value Adjustment (MVA):
Some Athene MYGA contracts include an MVA that adjusts the surrender
value based on changes in interest rates since issue. If rates have risen,
the MVA is negative (reduces surrender value). If rates have fallen, the
MVA is positive (increases surrender value). The MVA applies only to
amounts subject to surrender charges.""",
            {"source": "surrender_and_liquidity_guide.pdf", "page": 1, "doc_type": "guide"},
        ),

        # ── Agent Product Comparison ──────────────────────────────────────
        (
            """Athene Product Comparison — Agent Reference Guide

FIA vs. MYGA: When to Recommend Which

Fixed Indexed Annuity (FIA):
- Best for: Clients with a 7–15 year time horizon who want principal
  protection AND the potential to earn more than a fixed rate by
  participating in index-linked crediting.
- Index options: S&P 500 (annual point-to-point with cap), Bloomberg
  U.S. Aggregate Bond Index, MSCI EAFE.
- Rider availability: GLWB, Enhanced Death Benefit, Return of Premium.
- Minimum premium: Typically $10,000–$25,000.
- Ideal client profile: Age 55–70, moderate risk tolerance, accumulation
  or income planning goal, comfortable with variable (but floored) returns.

Multi-Year Guaranteed Annuity (MYGA):
- Best for: Clients who want certainty of a specific rate for a defined
  term — no surprises, no market linkage.
- Best positioned as a CD alternative or bond ladder substitute.
- Minimum premium: Typically $5,000–$10,000.
- Ideal client profile: Age 50–75, conservative risk tolerance,
  near-term income or CD rollover situation, prioritizes simplicity.

Key Differentiators:
  Feature               FIA                     MYGA
  ─────────────────────────────────────────────────────
  Interest method       Index-linked (0% floor)  Declared fixed rate
  Upside potential      Moderate (capped)         None — rate is fixed
  Predictability        Lower                     High
  Rider availability    Yes (GLWB, etc.)          No
  Typical rate vs MYGA  Potentially higher        Guaranteed from day 1

Positioning Both Products Together:
A common strategy is to split a client's retirement assets between a MYGA
(for a guaranteed rate on a portion) and an FIA with a GLWB (for
long-term income floor). This laddering approach balances certainty and
upside potential.""",
            {"source": "agent_product_comparison.pdf", "page": 1, "doc_type": "guide"},
        ),


        # ── Annuitization Options ─────────────────────────────────────────
        (
            """Annuitization and Payout Options

Annuitization converts the accumulated account value into a guaranteed
stream of income payments. Once annuitization begins, the decision is
generally irrevocable -- the owner exchanges the lump-sum account value
for a series of income payments.

Common Annuitization Options:
1. Life Only (Straight Life): Payments continue for the annuitant's
   lifetime and stop at death. Produces the highest monthly payment of
   any option, but no residual value passes to heirs.
2. Life with Period Certain: Payments are guaranteed for the longer of
   the annuitant's lifetime or a specified period (e.g., 10 or 20 years).
   If the annuitant dies within the period certain, payments continue to
   the beneficiary for the remainder of the period.
3. Joint and Survivor: Payments continue as long as either the annuitant
   or the joint annuitant (typically a spouse) is alive. A 100% Joint and
   Survivor option continues full payments to the survivor; a 50% J&S
   option reduces payments by 50% after the first death.
4. Period Certain Only: Payments are made for a fixed number of years
   (e.g., 20 years) regardless of whether the annuitant is alive.

Annuitization vs. GLWB Withdrawals:
Most modern FIA clients with income needs use the GLWB rider rather than
annuitization, because:
- GLWB preserves account value access (annuitization does not).
- GLWB allows the remaining account value to pass to heirs.
- GLWB income can increase if the account value grows.
Annuitization may produce higher income than GLWB if the account value
is very large relative to the Benefit Base, or if the client has no
legacy goals and prioritizes maximum monthly income.

Exclusion Ratio (Non-Qualified Contracts):
When a non-qualified annuity is annuitized, each payment consists partly
of a return of principal (tax-free) and partly of earnings (taxable).
The exclusion ratio = Total investment in contract divided by Expected return.""",
            {"source": "annuitization_options_guide.pdf", "page": 1, "doc_type": "guide"},
        ),
    ]

    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in raw_docs
    ]

    return documents
