// Provider catalogs: probe each provider's model list in its own wire,
// then build the one catalog document the entry registers and `--list`
// prints. The ratings table is the truth for free flags; the probe is the
// truth for what a provider actually lists. A provider whose probe fails
// is left out of the catalog and out of routing.

import { isRecord } from './echo.ts';
import type { Brief, ProviderEntry } from './brief.ts';

export interface ListedModel {
  id: string;
  free: boolean;
  contextLength: number | null;
}

export interface ProbeResult {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: ProviderEntry['wire'];
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const PROBE_TIMEOUT_MS = 5_000;

function authHeaders(provider: ProviderEntry): Record<string, string> {
  const headers: Record<string, string> = {};
  if (provider.auth === 'bearer') headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === 'x-api-key') headers['x-api-key'] = provider.key;
  else if (provider.auth === 'x-goog-api-key') headers['x-goog-api-key'] = provider.key;
  return headers;
}

function catalogUrl(envUrl: string, provider: ProviderEntry): string {
  const route = provider.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
  return `${envUrl}${provider.path}${route}`;
}

function parseListing(provider: ProviderEntry, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const raw: unknown[] =
    provider.wire === 'gemini'
      ? Array.isArray(body.models)
        ? body.models
        : []
      : Array.isArray(body.data)
        ? body.data
        : [];
  const models: ListedModel[] = [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    let id: unknown = item.id;
    if (provider.wire === 'gemini' && typeof item.name === 'string') {
      id = item.name.startsWith('models/') ? item.name.slice('models/'.length) : item.name;
    }
    if (typeof id !== 'string' || id === '') continue;
    models.push({
      id,
      free: item.free === true,
      contextLength: typeof item.context_length === 'number' ? item.context_length : null,
    });
  }
  return models;
}

async function probeOnce(envUrl: string, provider: ProviderEntry): Promise<ProbeResult> {
  try {
    const res = await fetch(catalogUrl(envUrl, provider), {
      headers: authHeaders(provider),
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!res.ok) return { provider, ok: false, models: [] };
    const models = parseListing(provider, await res.json());
    return { provider, ok: models.length > 0, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/** Probe one provider; a failed probe is retried once before giving up. */
export async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<ProbeResult> {
  const first = await probeOnce(envUrl, provider);
  if (first.ok) return first;
  return probeOnce(envUrl, provider);
}

/** Probe every provider in the brief, in parallel (catalog reads are free). */
export async function probeAll(envUrl: string, brief: Brief): Promise<ProbeResult[]> {
  return Promise.all(brief.providers.map((provider) => probeProvider(envUrl, provider)));
}

/**
 * The register/`--list` catalog: one entry per answered provider, base_url
 * as the provider's bare path (the harness normalizes a full URL to the
 * same suffix; the bare path is identical in live and replay), models in
 * listing order, free flags from the ratings table.
 */
export function buildCatalog(brief: Brief, probes: ProbeResult[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const provider = probe.provider;
    const models: CatalogModel[] = probe.models.map((listed) => {
      const row = brief.rowByKey.get(`${provider.id}/${listed.id}`);
      return {
        id: listed.id,
        free: row !== undefined ? row.free : listed.free,
        context_length: row !== undefined ? row.contextLength : (listed.contextLength ?? 131072),
      };
    });
    providers.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.keyEnv,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}

/** The flat free-id list the /v1/models route answers with (one per provider+model). */
export function freeModelIds(catalog: Catalog): string[] {
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) if (model.free) ids.push(model.id);
  }
  return ids;
}
