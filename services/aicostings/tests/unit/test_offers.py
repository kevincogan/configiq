# SPDX-License-Identifier: Apache-2.0

"""Complete-offer contract, collectors and snapshot lifecycle."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from tools.api_service.offers import RentalOffer, assemble_offers, load_curated_offers
from tools.api_service.scrapers.offer_collectors import parse_coreweave_prices, parse_gcp_a2_prices, snapshot_from_rates


def offer(**overrides):
    return RentalOffer.model_validate({
        "id": "test", "system_id": "l40s", "provider": "aws", "provider_region": "aws.us-east-1",
        "instance_name": "g6e.xlarge", "gpu_count": 1, "hourly_cost": 2.0, "rate_kind": "on_demand",
        "source": {"kind": "automated", "label": "Test provider", "url": "https://example.com/pricing"},
        **overrides,
    })


def test_catalogue_preserves_full_servers_not_card_prices():
    data = load_curated_offers()
    assert len(data.rented) == 28
    assert len(data.purchased) == 19
    server = next(record for record in data.purchased if record.id == "l40s:server:1")
    assert server.purchase_price == 35000
    assert server.installation_cost == 2500
    assert server.price_basis == "complete_server"
    assert all(record.source.kind == "curated" and record.source.retrieved_at is None for record in data.purchased)


@pytest.mark.parametrize("field,value", [("gpu_count", 0), ("gpu_count", 1.5), ("hourly_cost", 0),
                                        ("hourly_cost", float("inf")), ("hourly_cost", True), ("currency", "EUR"),
                                        ("rate_basis", "gpu_hour")])
def test_contract_rejects_incomplete_or_incompatible_prices(field, value):
    with pytest.raises(ValidationError):
        offer(**{field: value})


def test_successful_snapshot_removes_old_offer_without_reviving_curated_price():
    curated = load_curated_offers().model_dump(mode="json")
    snapshot = {"offers": [], "collected_at": datetime.now(UTC).isoformat(),
                "scope": {"instances": ["g6e.xlarge"], "regions": ["aws.us-east-1"]}}
    data = assemble_offers(curated, {"aws": snapshot}, {})
    assert not any(record.instance_name == "g6e.xlarge" for record in data.rented)
    assert any(record.instance_name == "g6e.48xlarge" for record in data.rented)


def test_failed_refresh_retains_last_good_price_but_marks_it_stale():
    record = offer()
    snapshot = {"offers": [record.model_dump(mode="json")], "collected_at": datetime.now(UTC).isoformat(),
                "scope": {"instances": [record.instance_name], "regions": [record.provider_region]}}
    data = assemble_offers(None, {"aws": snapshot}, {"offers.aws": {"message": "Failed"}})
    assert data.rented[0].hourly_cost == 2
    assert data.rented[0].source.stale
    assert data.sources["aws"].state == "stale"
    snapshot["collected_at"] = (datetime.now(UTC) - timedelta(hours=26)).isoformat()
    assert assemble_offers(None, {"aws": snapshot}, {}).rented[0].source.stale


def test_no_automatic_snapshot_means_explicit_curated_or_unconfigured_not_fake_live():
    data = assemble_offers(load_curated_offers().model_dump(mode="json"), {}, {})
    assert data.sources["gcp"].state == "unconfigured"
    assert all(record.source.kind == "curated" for record in data.rented)
    failed = assemble_offers(load_curated_offers().model_dump(mode="json"), {}, {"offers.aws": {}})
    assert not any(record.provider == "aws" for record in failed.rented)


def test_rate_projection_retains_whole_instance_and_excludes_wrong_gpu_variants():
    records = [{"system_id": "a100_sxm", "provider_region": "aws.us-east-1", "instance_name": name,
                "gpus_per_instance": 8, "instance_hourly_cost": 32, "on_demand": 4,
                "source_url": "https://example.com/aws.json", "currency": "USD", "unit": "Hrs"}
               for name in ("p4d.24xlarge", "p4de.24xlarge")]
    data = snapshot_from_rates("aws", records)
    assert len(data["offers"]) == 1
    assert data["offers"][0]["instance_name"] == "p4de.24xlarge"
    assert data["offers"][0]["hourly_cost"] == 32
    assert data["offers"][0]["rate_basis"] == "instance_hour"


def coreweave_row(name="NVIDIA HGX B200", price="$68.80", inference="$8.60"):
    cells = [name, "8", "180", "128", "2,048", "61.44", price, "$34.11", inference]
    return ('<h5><strong>REGION: NORTH AMERICA</strong></h5>'
            '<div class="table-row-v2 w-dyn-item kubernetes-gpu-pricing"><div class="table-grid">'
            + ''.join(f'<div class="table-v2-cell"><div>{cell}</div></div>' for cell in cells)
            + '</div></div><p>Pricing based on GPUs is exclusively applicable to inference platform customers.</p>')


def test_coreweave_keeps_instance_spot_and_restricted_per_gpu_prices_distinct():
    offers = parse_coreweave_prices(coreweave_row())
    assert [(record.gpu_count, record.hourly_cost, record.eligibility) for record in offers] == [
        (8, 68.8, "standard"), (8, 34.11, "standard"), (1, 8.6, "inference_platform")]
    assert offers[1].rate_kind == "spot"
    assert all(record.source.kind == "automated" and record.source.price_date is None for record in offers)
    assert parse_coreweave_prices(coreweave_row(price="Contact sales"))[0].rate_kind == "spot"


def test_coreweave_fails_on_layout_or_eligibility_changes():
    with pytest.raises(ValueError):
        parse_coreweave_prices('<p>Contact sales</p>')
    with pytest.raises(ValueError):
        parse_coreweave_prices(coreweave_row().replace("exclusively applicable", "changed terms"))
    with pytest.raises(ValueError):
        parse_coreweave_prices(coreweave_row(name="NVIDIA L40"))  # not L40S


def gcp_sku(description, price, unit="h"):
    return {"skuId": description, "description": description, "serviceRegions": ["us-central1"],
            "category": {"usageType": "OnDemand"}, "pricingInfo": [{
                "effectiveTime": "2026-10-01T00:00:00Z", "pricingExpression": {
                    "usageUnit": unit, "tieredRates": [{"startUsageAmount": 0, "unitPrice": {
                        "currencyCode": "USD", "units": str(int(price)), "nanos": round((price - int(price)) * 1e9)}}]}}]}


def test_gcp_requires_all_machine_components_and_rejects_ambiguous_prices():
    skus = [gcp_sku("NVIDIA A100 80GB GPU", 5), gcp_sku("A2 Ultra Instance Core", .1),
            gcp_sku("A2 Ultra Instance RAM", .01, "GiBy.h"), gcp_sku("Local SSD", .02, "GiBy.h")]
    offers = parse_gcp_a2_prices(skus, ["us-central1"])
    assert [record.gpu_count for record in offers] == [1, 2, 4, 8]
    assert offers[0].hourly_cost == pytest.approx(5 + 12 * .1 + 170 * .01 + 375 * .02)
    for invalid in [skus[:-1], [*skus, skus[0]]]:
        with pytest.raises(ValueError):
            parse_gcp_a2_prices(invalid, ["us-central1"])
    future_price = gcp_sku("NVIDIA A100 80GB GPU", 100)["pricingInfo"][0]
    future_price["effectiveTime"] = "2100-01-01T00:00:00Z"
    skus[0]["pricingInfo"].append(future_price)
    assert parse_gcp_a2_prices(skus, ["us-central1"])[0].hourly_cost == offers[0].hourly_cost


@pytest.mark.asyncio
async def test_offers_endpoint_is_additive_and_preserves_gpu_price_meaning(monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from tools.api_service.app import app, store

    monkeypatch.setattr(store, "get_curated_offers", lambda: load_curated_offers().model_dump(mode="json"))
    monkeypatch.setattr(store, "get_offer_snapshots", dict)
    monkeypatch.setattr(store, "get_scrape_errors", dict)
    monkeypatch.setattr(store, "get_scrape_times", dict)
    monkeypatch.setattr(store, "get_all_cloud_rates", dict)
    monkeypatch.setattr(store, "get_all_hardware_costs", lambda: {"h100_sxm": {
        "new_usd": 39750, "new_node_usd": 318000, "install_cost_node": 22300}})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/offers")
        legacy = await client.get("/systems?include=hardware")
        # Expiration must not bring back a superseded manual AWS price; a
        # successful curated reload is not a missing provider snapshot.
        monkeypatch.setattr(store, "get_scrape_times", lambda: {"offers.aws": "old", "offers.curated": "now"})
        expired = (await client.get("/offers")).json()
    assert response.status_code == 200
    assert response.json()["rented"][0]["rate_basis"] == "instance_hour"
    hardware = next(record for record in legacy.json()["systems"] if record["id"] == "h100_sxm")["hardware_cost"]
    assert hardware["new_usd"] == 39750
    assert hardware["new_node_usd"] == 318000
    assert not any(record["provider"] == "aws" for record in expired["rented"])
    assert expired["sources"]["aws"]["state"] == "unavailable"
    assert expired["sources"]["curated"]["message"] == "Manually maintained, dated estimates; not live quotes."


@pytest.mark.asyncio
async def test_startup_serves_curated_data_without_waiting_for_network_and_cancels_cleanly(monkeypatch):
    import asyncio

    from tools.api_service import app as api

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_collection():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    store = MagicMock()
    load = AsyncMock()
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "load_device_names_from_perf_data", dict)
    monkeypatch.setattr(api, "AsyncIOScheduler", MagicMock())
    monkeypatch.setattr(api, "job_load_hardware_costs", load)
    monkeypatch.setattr(api, "job_scrape_models", AsyncMock())
    monkeypatch.setattr(api, "job_scrape_cloud_rates", slow_collection)
    async with api.lifespan(api.app):
        load.assert_awaited_once()
        await asyncio.wait_for(started.wait(), timeout=1)
        store.close.assert_not_called()
    assert cancelled.is_set()
    store.close.assert_called_once()
