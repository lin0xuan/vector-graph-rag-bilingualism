import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from vector_graph_rag_bilingualism import ChinesePipeline, SnapshotStore, EnglishConfig
from vector_graph_rag_bilingualism import cli
from vector_graph_rag_bilingualism.english import EnglishUpstreamPipeline
from vector_graph_rag_bilingualism.milvus import (
    DEFAULT_MILVUS_URI, MilvusMirror, standalone_uri, connect_standalone,
)
from vector_graph_rag_bilingualism.migration import sync_chinese, migrate_english
from helpers import FakeEmbedding, FakeModel, corpus


@pytest.fixture(autouse=True)
def clean_endpoint(monkeypatch):
    monkeypatch.delenv('VGRAG_MILVUS_URI', raising=False)


@pytest.mark.parametrize('uri', ['./old.db', '/tmp/index.db', 'file:///index.db',
                                 'http://user:secret@localhost:19530',
                                 'http://localhost:19530?token=secret'])
def test_reject_local_files_and_embedded_credentials(uri):
    with pytest.raises(ValueError):
        standalone_uri(uri)
    # Rejected before importing upstream or constructing any model clients.
    with pytest.raises(ValueError):
        EnglishUpstreamPipeline.connect(None, 'en', EnglishConfig({'milvus_uri': uri}))


def test_endpoint_override_and_default(monkeypatch):
    assert standalone_uri() == DEFAULT_MILVUS_URI
    monkeypatch.setenv('VGRAG_MILVUS_URI', 'http://localhost:19531/')
    assert standalone_uri('http://localhost:19530') == 'http://localhost:19531'


def test_english_passes_shared_endpoint_and_token_without_lite(monkeypatch):
    from vector_graph_rag_bilingualism import english
    captured = {}
    class Settings:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def model_dump(self, **kwargs):
            return dict(captured)
    storage = SimpleNamespace(client=Mock(), entity_collection='e', relation_collection='r', passage_collection='p')
    def construct(*, settings):
        return SimpleNamespace(_store=storage, settings=settings)
    monkeypatch.setitem(sys.modules, 'vector_graph_rag', SimpleNamespace(VectorGraphRAG=construct))
    monkeypatch.setitem(sys.modules, 'vector_graph_rag.config', SimpleNamespace(Settings=Settings))
    monkeypatch.setattr(english, 'verify_upstream_version', lambda: None)
    monkeypatch.setenv('VGRAG_MILVUS_URI', 'http://127.0.0.1:19531')
    monkeypatch.setenv('VGRAG_MILVUS_TOKEN', 'fixture-secret')
    config = EnglishConfig({'embedding_provider': 'openai', 'embedding_model': 'fixture',
                            'embedding_dimension': 16, 'llm_model': 'fixture',
                            'final_top_k': 3, 'collection_prefix': 'fixture'})
    pipeline = EnglishUpstreamPipeline.connect(None, 'en', config)
    assert captured['milvus_uri'] == 'http://127.0.0.1:19531'
    assert captured['milvus_token'] == 'fixture-secret'
    assert captured['reranker_model'] == 'jev-1.13.0'
    assert 'fixture-secret' not in json.dumps(pipeline.identity)
    captured.clear()
    EnglishUpstreamPipeline.connect(None, 'en', config, use_jev=False)
    assert 'reranker_model' not in captured


def test_default_chinese_uses_standalone_and_does_not_fallback(tmp_path, monkeypatch):
    mirror = Mock()
    factory = Mock(return_value=mirror)
    monkeypatch.setattr(MilvusMirror, 'connect', factory)
    model = FakeModel()
    pipeline = ChinesePipeline(SnapshotStore(tmp_path), 'zh', FakeEmbedding(), model)
    assert pipeline.mirror is mirror
    snapshot = pipeline.build(corpus())
    assert mirror.ensure.call_args.args[0] == snapshot
    mirror.validate.side_effect = RuntimeError('server unavailable')
    before = model.calls.copy()
    with pytest.raises(RuntimeError, match='unavailable'):
        pipeline.query('谁负责审批？')
    assert model.calls == before
    local = ChinesePipeline(pipeline.store, 'zh', FakeEmbedding(), model, local_search=True)
    assert pipeline.fingerprint() != local.fingerprint()


def test_cli_defaults_to_server(tmp_path, monkeypatch):
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(MilvusMirror, 'connect', factory)
    service = {'base_url': 'http://localhost:8081/v1', 'model': 'fixture', 'key_env': 'FIXTURE_KEY'}
    config = {'language': 'zh', 'namespace': 'zh', 'data_dir': str(tmp_path), 'use_jev': False,
              'embedding': {'dimension': 16, 'service': service}, 'models': {'default': service}}
    result = cli.connect(config)
    assert result.mirror is factory.return_value
    assert result.reranker is None
    factory.assert_called_once_with(None)


def test_chinese_cli_defaults_to_typesafe_jev(tmp_path, monkeypatch):
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(MilvusMirror, 'connect', factory)
    created = {}

    class Marker:
        def __init__(self, **options):
            created.update(options)

        def identity(self):
            return {"provider": "jev"}

    monkeypatch.setattr('vector_graph_rag_bilingualism.jev.JevReranker', Marker)
    service = {'base_url': 'http://localhost:8081/v1', 'model': 'fixture', 'key_env': 'FIXTURE_KEY'}
    config = {'language': 'zh', 'namespace': 'zh', 'data_dir': str(tmp_path),
              'embedding': {'dimension': 16, 'service': service}, 'models': {'default': service}}
    result = cli.connect(config)
    assert created == {'service': 'typesafe', 'model': 'jev-1.13.0'}
    assert result.reranker is result.relation_reranker
    assert result.config.profile == 'zh_coverage_v2'


def test_chinese_sync_has_no_model_dependency_and_reuses_snapshot(tmp_path, monkeypatch, capsys):
    store = SnapshotStore(tmp_path)
    local = ChinesePipeline(store, 'zh', FakeEmbedding(), FakeModel(), local_search=True)
    sid = local.build(corpus())
    mirror = Mock()
    mirror.name.side_effect = MilvusMirror.name
    monkeypatch.setattr(MilvusMirror, 'connect', Mock(return_value=mirror))
    config = {'language': 'zh', 'namespace': 'zh', 'data_dir': str(tmp_path)}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))  # No embedding/model config or credentials needed.
    cli.main(['--config', str(path), 'sync-milvus'])
    assert json.loads(capsys.readouterr().out)['snapshot'] == sid
    assert mirror.ensure.call_args.args[0] == sid
    assert store.current('zh') == sid
    mirror.client.close.assert_called_once()
    with pytest.raises(ValueError):
        sync_chinese({**config, 'namespace': 'another'}, sid)


def test_existing_mirror_is_validated_never_overwritten():
    client = Mock()
    client.has_collection.return_value = True
    mirror = MilvusMirror(client)
    mirror.validate = Mock(side_effect=ValueError('changed'))
    mirror.publish = Mock()
    with pytest.raises(ValueError, match='changed'):
        mirror.ensure('sid', {}, [])
    mirror.publish.assert_not_called()


def test_connection_failure_is_redacted(monkeypatch):
    import pymilvus
    monkeypatch.setattr(pymilvus, 'MilvusClient', Mock(side_effect=Exception('secret-token')))
    with pytest.raises(RuntimeError, match='Standalone') as error:
        connect_standalone()
    assert 'secret-token' not in str(error.value)


def test_english_copies_vectors_preserves_source_and_checks_existing_target(tmp_path, monkeypatch):
    import pymilvus
    from vector_graph_rag_bilingualism.config import UPSTREAM_COMMIT
    from vector_graph_rag_bilingualism.models import Passage, digest
    from vector_graph_rag_bilingualism import migration
    settings = {'milvus_uri': 'old.db', 'collection_prefix': 'old',
                'entity_collection': 'entities', 'relation_collection': 'relations',
                'passage_collection': 'passages', 'milvus_index_type': 'AUTOINDEX',
                'embedding_model': 'frozen-model'}
    content = {f'old_{kind}': [{'id': kind, 'text': kind, 'vector': [1.0, 0.0],
                               'entity_ids': ['entities']}]
               for kind in ('entities', 'relations', 'passages')}
    source = Mock()
    target = Mock()
    target_content = {}
    target.has_collection.side_effect = lambda name: name in target_content
    target.create_collection.side_effect = lambda **kw: target_content.setdefault(kw['collection_name'], [])
    target.insert.side_effect = lambda **kw: target_content[kw['collection_name']].extend(kw['data'])
    target.drop_collection.side_effect = lambda name: target_content.pop(name)
    monkeypatch.setattr(pymilvus, 'MilvusClient', Mock(return_value=source))
    monkeypatch.setattr(migration, 'connect_standalone', Mock(return_value=target))
    def dump(client, name):
        return content[name] if client is source else target_content[name]
    monkeypatch.setattr(migration, 'dump_collection', dump)
    monkeypatch.setattr(migration, 'collection_hashes',
                        lambda c, names: {n: digest(dump(c, n)) for n in names})
    store = SnapshotStore(tmp_path / 'source')
    sid = store.create('en', 'en', [Passage('passages', 'doc', 'passages', 'en', 'doc')], [[1.0, 0.0]], {},
                       {'commit': UPSTREAM_COMMIT, 'settings': settings},
                       external={'collections': {n: digest(rows) for n, rows in content.items()}})
    store.activate('en', sid)
    source_file = tmp_path / 'old.db'
    source_file.touch()
    config = {'language': 'en', 'namespace': 'en', 'data_dir': str(tmp_path / 'target'),
              'upstream': {'milvus_uri': DEFAULT_MILVUS_URI, 'collection_prefix': 'new',
                           'embedding_model': 'frozen-model'}}
    result = migrate_english(config, store.directory, str(source_file), sid)
    new_store = SnapshotStore(tmp_path / 'target')
    manifest, _ = new_store.load(result['snapshot'])
    assert manifest['config']['settings']['milvus_uri'] == DEFAULT_MILVUS_URI
    assert new_store.current('en') == result['snapshot']
    assert store.current('en') == sid
    source.insert.assert_not_called()
    assert target_content['new_relations'] == content['old_relations']
    # Idempotent retry checks content rather than overwriting it.
    assert migrate_english(config, store.directory, str(source_file), sid) == result
    assert target.create_collection.call_count == 3
    target_content['new_relations'] = []
    with pytest.raises(ValueError, match='different data'):
        migrate_english(config, store.directory, str(source_file), sid)
    target.drop_collection.assert_not_called()
    # A failed fresh copy must not activate a snapshot or affect existing collections.
    before = dict(target_content)
    config['upstream']['collection_prefix'] = 'failed'
    target.insert.side_effect = RuntimeError('interrupted transfer')
    with pytest.raises(RuntimeError, match='interrupted'):
        migrate_english(config, store.directory, str(source_file), sid)
    target.drop_collection.assert_called_once_with('failed_entities')
    assert target_content == before
    assert new_store.current('en') == result['snapshot']
    assert store.current('en') == sid
