# Analytics log storage (Supabase Postgres, via PostgREST)
create table if not exists public.logs (
  id uuid primary key,
  session_id text,
  timestamp timestamptz,
  query text,
  response text,
  sources text,
  feedback smallint
);

alter table public.logs enable row level security;

create index if not exists logs_timestamp_idx on public.logs (timestamp desc);