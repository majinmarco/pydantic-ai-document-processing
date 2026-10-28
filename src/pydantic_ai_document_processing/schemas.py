from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentType = Literal[
    "invoice",
    "packing_slip",
    "purchase_order",
    "quote",
    "shipping_order",
    "work_order",
]


class DocumentClassification(BaseModel):
    document_type: DocumentType


class Address(BaseModel):
    line1: str | None = None
    line2: str | None = None
    city: str | None = None
    region: str | None = Field(
        default=None,
        description="State, province, or county. Not the city or country.",
    )
    postal_code: str | None = Field(
        default=None,
        description="ZIP or postal code as printed.",
    )
    country: str | None = None


class Bank(BaseModel):
    bank_name: str | None = None
    iban: str | None = Field(
        default=None,
        description="IBAN if present; do not put a local account number here.",
    )
    bic: str | None = Field(
        default=None,
        description="BIC/SWIFT code, not the account number.",
    )
    account_number: str | None = Field(
        default=None,
        description="Domestic account number when no IBAN is printed.",
    )
    routing_number: str | None = Field(
        default=None,
        description="Bank routing/sort code (e.g. ABA, sort code).",
    )


class Party(BaseModel):
    name: str | None = None
    address: Address | None = None
    phone: str | None = None
    email: str | None = None
    website: str | None = None
    tax_id: str | None = Field(
        default=None,
        description="VAT, GST, EIN, or other tax registration as printed, including prefix.",
    )
    code: str | None = Field(
        default=None,
        description="Vendor, customer, or carrier account number on the document, not a country code.",
    )
    bank: Bank | None = None


class Parties(BaseModel):
    issuer: Party | None = Field(
        default=None,
        description="Party that issued the document (seller on invoices/quotes; buyer on purchase orders).",
    )
    bill_to: Party | None = Field(
        default=None,
        description="Buyer billed for payment. Distinct from ship-to / deliver-to / consignee.",
    )
    ship_to: Party | None = Field(
        default=None,
        description="Delivery recipient when different from bill-to.",
    )
    vendor: Party | None = Field(
        default=None,
        description="Supplier being ordered from on a purchase order. Not the document issuer.",
    )
    deliver_to: Party | None = Field(
        default=None,
        description="PO ship-to / delivery location. Distinct from vendor.",
    )
    consignee: Party | None = Field(
        default=None,
        description="Shipment recipient on a shipping order. Distinct from the carrier.",
    )
    carrier: Party | None = Field(
        default=None,
        description="Transport company as a party (name/code). If only a carrier name string is printed, put it in refs.carrier.",
    )


class DocumentDates(BaseModel):
    order_date: date | None = Field(
        default=None,
        description="Date the related sales/purchase order was placed, not the document date.",
    )
    due_date: date | None = Field(
        default=None,
        description="Due date as printed: payment due on invoices; planned completion on work orders.",
    )
    delivery_date: date | None = Field(
        default=None,
        description="Scheduled or promised delivery date, not pickup or latest-by.",
    )
    valid_until: date | None = Field(
        default=None,
        description="Quote expiry date. Put validity prose (e.g. 'valid for 30 days') in extra.",
    )
    pickup_date: date | None = Field(
        default=None,
        description="Date the carrier collects the shipment, not the delivery date.",
    )
    deliver_by: date | None = Field(
        default=None,
        description="Latest allowed delivery date on a shipping order, not the scheduled delivery_date.",
    )
    end_date: date | None = Field(
        default=None,
        description="Actual or recorded finish date on a work order, distinct from due_date.",
    )


class DocumentRefs(BaseModel):
    order_id: str | None = Field(
        default=None,
        description="Seller sales-order number (e.g. SO…), not this document's number.",
    )
    customer_id: str | None = Field(
        default=None,
        description="Buyer account id; may match bill-to.code.",
    )
    po_number: str | None = Field(
        default=None,
        description="Buyer's purchase-order number when this document is not itself a PO.",
    )
    carrier: str | None = Field(
        default=None,
        description="Carrier name as a string (packing slip / PO). Use parties.carrier when the carrier is a full party.",
    )
    account_number: str | None = Field(
        default=None,
        description="Vendor account number on a PO header; often matches vendor.code. Not a bank account.",
    )
    requested_by: str | None = Field(
        default=None,
        description="Person who requested the purchase order.",
    )


class LineItem(BaseModel):
    sku: str | None = Field(
        default=None,
        description="Item, product, or operation code, not the human description.",
    )
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = Field(
        default=None,
        description="Price per unit before discount and tax. May be 0.00 on packing slips and work orders.",
    )
    discount_rate: Decimal | None = Field(
        default=None,
        description="Line discount as a decimal fraction (0.10 = 10%), not a money amount.",
    )
    tax_rate: Decimal | None = Field(
        default=None,
        description="Line tax as a decimal fraction (0.20 = 20%), not a money amount.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measure (EA, KG, HR), not the quantity.",
    )
    line_total: Decimal | None = Field(
        default=None,
        description="Extended line amount after discount, usually before tax.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Leftover line facts that do not map to other fields "
            "(category, shipped, backordered, received, rejected, stocked, weight, hours, costs, planned/actual dates)."
        ),
    )


class TaxLine(BaseModel):
    rate: Decimal | None = Field(
        default=None,
        description="Tax rate as a decimal fraction (0.0786 = 7.86%), not a percent integer.",
    )
    base: Decimal | None = Field(
        default=None,
        description="Taxable amount this rate applies to, not the tax charged.",
    )
    amount: Decimal | None = Field(
        default=None,
        description="Tax charged for this rate (rate × base).",
    )
    name: str | None = Field(
        default=None,
        description="Tax label if printed (VAT, GST, sales tax).",
    )


class Totals(BaseModel):
    currency: str | None = None
    subtotal: Decimal | None = Field(
        default=None,
        description="Sum of line totals after line discounts, before freight and tax.",
    )
    gross: Decimal | None = Field(
        default=None,
        description="Goods total before freight and tax; often equal to subtotal.",
    )
    discount_total: Decimal | None = Field(
        default=None,
        description="Document-level discount amount, not a rate.",
    )
    freight: Decimal | None = Field(
        default=None,
        description="Shipping/freight/postage amount.",
    )
    taxes: list[TaxLine] | None = None
    tax_total: Decimal | None = Field(
        default=None,
        description="Sum of tax amounts, not the taxable base.",
    )
    total: Decimal | None = Field(
        default=None,
        description="Amount due: typically subtotal − discount + freight + tax.",
    )


class SummaryRow(BaseModel):
    label: str | None = Field(
        default=None,
        description="Row key as a snake_case identifier (packages, total_units, freight, variance), not display prose.",
    )
    value: str | int | Decimal | None = Field(
        default=None,
        description="Printed value; keep units inside the string when present (e.g. '11.53 kg').",
    )
    fmt: Literal["int", "text", "money", "number"] | None = Field(
        default=None,
        description="How to interpret value: int, text, money, or number.",
    )
    strong: bool | None = Field(
        default=None,
        description="True when the document emphasizes this row (bold / total line).",
    )


class Invoice(BaseModel):
    doc_type: Literal["invoice"] | None = None
    number: str | None = Field(
        default=None,
        description="Invoice number as printed, not order, PO, or customer id.",
    )
    issue_date: date | None = Field(
        default=None,
        description="Invoice date. Put order/due/delivery dates under dates.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code (e.g. GBP, USD).",
    )
    parties: Parties | None = None
    dates: DocumentDates | None = None
    refs: DocumentRefs | None = None
    items: list[LineItem] | None = None
    totals: Totals | None = None
    notes: str | None = Field(
        default=None,
        description="Printed remarks or terms text, not structured payment fields.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Leftover invoice facts that do not map to other fields (e.g. payment_terms).",
    )


class PackingSlip(BaseModel):
    doc_type: Literal["packing_slip"] | None = None
    number: str | None = Field(
        default=None,
        description="Delivery-note / packing-slip number, not the sales-order or PO number.",
    )
    issue_date: date | None = Field(
        default=None,
        description="Packing-slip / dispatch date.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code if printed; packing slips may still show zero prices.",
    )
    parties: Parties | None = None
    refs: DocumentRefs | None = None
    items: list[LineItem] | None = None
    summary: list[SummaryRow] | None = Field(
        default=None,
        description="Package/unit counts and similar dispatch totals, not invoice money totals.",
    )
    notes: str | None = Field(
        default=None,
        description="Printed delivery remarks, not carrier or order ids.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Leftover packing-slip facts that do not map to other fields.",
    )


class PurchaseOrder(BaseModel):
    doc_type: Literal["purchase_order"] | None = None
    number: str | None = Field(
        default=None,
        description="This PO's number as printed. Do not also copy it to refs.po_number.",
    )
    issue_date: date | None = Field(
        default=None,
        description="Purchase-order date. Put requested delivery under dates.delivery_date.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code (e.g. GBP, USD).",
    )
    parties: Parties | None = None
    dates: DocumentDates | None = None
    refs: DocumentRefs | None = None
    items: list[LineItem] | None = None
    totals: Totals | None = None
    notes: str | None = Field(
        default=None,
        description="Printed remarks. Put payment terms and vendor instructions in extra.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Leftover PO facts (e.g. payment_terms, instructions).",
    )


class Quote(BaseModel):
    doc_type: Literal["quote"] | None = None
    number: str | None = Field(
        default=None,
        description="Quote / quotation number as printed, not a customer id.",
    )
    issue_date: date | None = Field(
        default=None,
        description="Quote date. Put expiry in dates.valid_until.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code (e.g. GBP, USD).",
    )
    parties: Parties | None = None
    dates: DocumentDates | None = None
    refs: DocumentRefs | None = None
    items: list[LineItem] | None = None
    totals: Totals | None = None
    notes: str | None = Field(
        default=None,
        description="Courtesy or footer remarks, not the validity clause.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Leftover quote facts (e.g. validity prose). Put the expiry date in dates.valid_until.",
    )


class ShippingOrder(BaseModel):
    doc_type: Literal["shipping_order"] | None = None
    number: str | None = Field(
        default=None,
        description="Shipment / bill-of-lading number, not the sales-order number.",
    )
    issue_date: date | None = Field(
        default=None,
        description="Shipping-order date. Put pickup and deliver-by under dates.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code if printed; freight may appear only in summary.",
    )
    parties: Parties | None = None
    dates: DocumentDates | None = None
    refs: DocumentRefs | None = None
    items: list[LineItem] | None = None
    summary: list[SummaryRow] | None = Field(
        default=None,
        description="Packages, weight, freight terms, and freight amount rows.",
    )
    notes: str | None = Field(
        default=None,
        description="Handling / shipping instructions, not freight terms (those go in summary).",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Leftover shipping-order facts that do not map to other fields.",
    )


class WorkOrder(BaseModel):
    doc_type: Literal["work_order"] | None = None
    number: str | None = Field(
        default=None,
        description="Work-order number as printed, not a product SKU.",
    )
    issue_date: date | None = Field(
        default=None,
        description="Work-order date. Put planned due and actual end under dates.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code if printed; costs may appear only in summary or line extra.",
    )
    parties: Parties | None = None
    dates: DocumentDates | None = None
    items: list[LineItem] | None = Field(
        default=None,
        description="Operations / routing steps. sku is the operation code; hours and costs go in extra.",
    )
    summary: list[SummaryRow] | None = Field(
        default=None,
        description="Hours, planned/actual cost, and variance totals.",
    )
    notes: str | None = Field(
        default=None,
        description="Printed remarks, not product identity (that goes in extra.product).",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Leftover work-order facts, including product identity pairs (product, product_number, quantities).",
    )

class ProcessResult(BaseModel):
    document_type: DocumentType | None = None
    document_data: Invoice | PackingSlip | PurchaseOrder | Quote | ShippingOrder | WorkOrder | None = None
    success: bool
    error_message: str | None = None