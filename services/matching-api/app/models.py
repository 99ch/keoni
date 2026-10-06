"""Modèles SQLAlchemy pour la persistance des embeddings (pgvector).

EMBEDDING_DIM doit rester cohérent avec le modèle sentence-transformers
configuré (SENTENCE_MODEL) : intfloat/multilingual-e5-base produit des
vecteurs à 768 dimensions. Changer de modèle pour un modèle d'une autre
dimension nécessite une nouvelle migration Alembic.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class JobEmbedding(Base):
    __tablename__ = "job_embeddings"
    __table_args__ = (UniqueConstraint("job_id", name="ux_job_embeddings_job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Le vecteur d'embedding vit dans JobEmbeddingChunk (voir plus bas) --
    # même raison que pour CvEmbedding : une offre longue (plusieurs
    # sections concaténées par prepare_job()) tronquait silencieusement
    # l'unique embedding a la limite interne du modele (512 tokens).
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class JobEmbeddingChunk(Base):
    """Un embedding par fenêtre de texte d'offre (voir chunk_text() dans
    main.py), pas un seul par offre entière -- même principe que
    CvEmbeddingChunk. La recherche vectorielle compare chaque fenêtre
    d'offre à chaque fenêtre de CV et garde la distance minimale : la
    meilleure paire de fenêtres gagne, jamais une moyenne qui diluerait un
    match localisé (même logique que cross_encode_best() pour le rerank)."""

    __tablename__ = "job_embedding_chunks"
    __table_args__ = (
        UniqueConstraint("job_id", "chunk_index", name="ux_job_embedding_chunks_job_id_chunk_index"),
        Index(
            "ix_job_embedding_chunks_vector",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": "100"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)


class CvEmbedding(Base):
    __tablename__ = "cv_embeddings"
    __table_args__ = (
        UniqueConstraint("cv_id", name="ux_cv_embeddings_cv_id"),
        # Index GIN pour le canal de récupération "exact" (chevauchement de
        # compétences canoniques ROME, opérateur `&&`) qui complète la
        # recherche vectorielle -- voir /retrieve dans main.py.
        Index("ix_cv_embeddings_skills_canonical", "skills_canonical", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cv_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Le vecteur d'embedding lui-même vit dans CvEmbeddingChunk (voir
    # plus bas) : un seul embedding pour tout le texte d'un CV le
    # tronquait silencieusement à la limite interne du modèle (512
    # tokens pour intfloat/multilingual-e5-base, ~350-400 mots) sur les
    # CV longs -- même problème que cross_encode_best() a déjà résolu
    # pour le rerank en découpant en fenêtres (voir chunk_text()).
    # Libellés canoniques (taxonomie ROME, voir app/taxonomy.py::find_skills)
    # détectés dans le texte du CV au moment de l'indexation -- alimente le
    # canal de récupération "exact" en complément du canal sémantique.
    # ARRAY(Text), pas ARRAY(String) : la colonne réelle est TEXT[] (voir
    # db.py::_ensure_skills_canonical_column) -- un mismatch avec VARCHAR[]
    # fait échouer l'opérateur `&&` côté Postgres ("operator does not
    # exist: text[] && character varying[]"), observé en prod sur /retrieve.
    skills_canonical: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    # Texte extrait du CV au moment du réindex (avant embedding) -- réutilisé
    # par /retrieve pour éviter à /score de re-extraire (OCR/Tika) un fichier
    # déjà traité. Voir db.py::ensure_pgvector_schema.
    text_content: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    # Faits lus dans le vrai texte du CV (document_profile.py, portage
    # AI Real-Time, 2026-10-06), destinés à terme à remplacer les champs de
    # formulaire WordPress (cv.metadata) que prepare_cv() lit aujourd'hui
    # pour ces mêmes informations. Calculés et stockés ici (pilier
    # Extraction) ; leur branchement dans prepare_cv()/le scoring est le
    # travail du pilier Scoring (scoring.py), pas encore fait -- ces
    # colonnes sont donc écrites mais pas encore lues ailleurs.
    experience_years: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    contract_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    education_text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    language_terms: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class CvEmbeddingChunk(Base):
    """Un embedding par fenêtre de texte (voir chunk_text() dans main.py),
    pas un seul embedding par CV entier -- pour qu'un CV long ne soit pas
    silencieusement tronqué à la limite interne du modèle (voir le
    commentaire sur CvEmbedding ci-dessus). La recherche vectorielle
    (/retrieve, rank_with_pgvector) regroupe par cv_id et prend la
    distance minimale : un candidat est classé sur sa meilleure fenêtre,
    jamais une moyenne qui diluerait un match localisé."""

    __tablename__ = "cv_embedding_chunks"
    __table_args__ = (
        UniqueConstraint("cv_id", "chunk_index", name="ux_cv_embedding_chunks_cv_id_chunk_index"),
        Index(
            "ix_cv_embedding_chunks_vector",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": "100"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cv_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)


class JobScoringState(Base):
    """Dernier passage du scoring autonome pour une offre (voir
    run_autonomous_scoring_cycle() dans main.py) -- permet de savoir si une
    offre a besoin d'être rescorée sans tout reparcourir à chaque cycle :
    absente d'ici = jamais scorée, `last_scored_job_modified` périmé =
    l'offre a changé côté WordPress, `last_scored_cv_pool_version` périmé =
    le vivier de CV a avancé depuis ce dernier passage (nouveau CV ajouté,
    CV existant réindexé) même si l'offre elle-même n'a pas bougé."""

    __tablename__ = "job_scoring_state"

    job_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Valeur brute renvoyée par GET /keoni/v1/job/{id} (`updated_at`) --
    # comparaison lexicographique suffisante sur son format Y-m-d H:i:s.
    last_scored_job_modified: Mapped[str] = mapped_column(String(32), nullable=False, server_default="")
    last_scored_cv_pool_version: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
