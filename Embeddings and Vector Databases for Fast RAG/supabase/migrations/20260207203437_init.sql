-- Create a dedicated schema for extensions
CREATE SCHEMA IF NOT EXISTS extensions;

-- Enable pgvector in the extensions schema
create extension if not exists vector with schema extensions;

-- Ensure the extensions schema is in the search path
-- (Supabase usually already includes this, but we make it explicit)
alter database postgres
set search_path = "$user", public, extensions;

-- 4. Create the chunks table
create table if not exists public.chunks (
    id bigserial primary key,
    content text not null,
    metadata jsonb,
    embedding vector(1024),
    fts tsvector generated always as (
        to_tsvector('english', content)
    ) stored
);

-- Vector similarity search (HNSW, cosine distance)
create index if not exists chunks_embedding_hnsw
on public.chunks
using hnsw (embedding vector_cosine_ops);

-- Full-text search
create index if not exists chunks_fts_gin
on public.chunks
using gin (fts);

-- Metadata filtering
create index if not exists chunks_metadata_gin
on public.chunks
using gin (metadata);

-- Vector similarity search function
create or replace function match_chunks(
    query_embedding vector(1024),
    match_count int default 5
)
returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        chunks.id,
        chunks.content,
        chunks.metadata,
        1 - (chunks.embedding <=> query_embedding) as similarity
    from chunks
    where chunks.embedding is not null
    order by chunks.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- Enable Row Level Security
alter table public.chunks enable row level security;