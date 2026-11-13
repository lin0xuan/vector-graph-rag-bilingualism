import re
from dataclasses import replace
import pytest
from helpers import FakeEmbedding, FakeModel, corpus
from vector_graph_rag_bilingualism import ChinesePipeline, ChineseConfig, SnapshotStore, Passage, Trace
from vector_graph_rag_bilingualism.retrieval import Retriever


def make(tmp_path, model=None, passages=None, **kwargs):
    pipeline = ChinesePipeline(SnapshotStore(tmp_path), 'repair', FakeEmbedding(), model or FakeModel(), **kwargs)
    pipeline.build(passages or corpus())
    return pipeline


@pytest.mark.parametrize('prefix', ['1. ', '2、', '(3) ', '（4）', '**5.** ', '- 6. '])
def test_numbered_lists_are_not_facts(tmp_path, prefix):
    p = make(tmp_path)
    p.model.call = lambda *args: {'ok': True, 'issues': [], 'gaps': []}
    assert p._verify(prefix + '张某负责审批。[p1]', {'passages': [{'id': 'p1', 'text': '张某负责审批。'}]}, Trace()) == []


def test_real_numeric_errors_still_rejected_and_multi_citations_work(tmp_path):
    p = make(tmp_path)
    p.model.call = lambda *args: {'ok': True, 'issues': [], 'gaps': []}
    payload = {'passages': [{'id': 'p1', 'text': '投资1.5万元。'}, {'id': 'p2', 'text': '期限30天。'}]}
    assert p._verify('1. 投资1.5万元，期限30天。[p1,p2]', payload, Trace()) == []
    assert 'numeric_copy_mismatch:2.5' in p._verify('2.5万元。[p1]', payload, Trace())
    assert 'numeric_copy_mismatch:31' in p._verify('1. 期限31天。[p2]', payload, Trace())


def test_short_ids_never_require_model_to_copy_hashes(tmp_path):
    docs = [replace(p, id=str(i)*64) for i,p in enumerate(corpus(),1)]
    model=FakeModel(); original=model.call
    def call(task,data,trace):
        if task=='select':
            assert all(re.fullmatch(r'p\d+',p['id']) and 'sentences' in p for p in data['candidates'])
        if task in ('answer','verify'):
            assert all(re.fullmatch(r'p\d+',p['id']) for p in data['passages'])
        return original(task,data,trace)
    model.call=call
    result=make(tmp_path,model,docs).query('谁负责审批，期限多久？')
    assert result.status=='complete'
    refs=re.findall(r'\[([^\]]+)\]',result.answer)
    assert refs and set(refs) <= {p.id for p in result.passages}
    assert all(len(ref)==64 for ref in refs)
    assert all(e.quote==next(p.text[e.start:e.end] for p in result.passages if p.id==e.passage_id) for e in result.evidence)


def test_technical_selection_failure_preserves_context_but_not_fake_coverage(tmp_path):
    model=FakeModel(); model.fail_select=True; original=model.call
    def call(task,data,trace):
        if task=='answer':
            assert data['missing_slots'] and not data['evidence']
            return {'answer':'\n'.join(p['text']+'['+p['id']+']' for p in data['passages'])}
        return original(task,data,trace)
    model.call=call
    p=make(tmp_path,model)
    retrieved=p.retrieve('谁负责审批，期限多久？')
    assert retrieved.passages and retrieved.missing_slots and not retrieved.evidence
    result=p.query('谁负责审批，期限多久？')
    assert result.answer and result.status=='partial'
    assert any(e['stage']=='selection_fallback' for e in result.trace.events)


def test_verified_gap_revisits_previously_rejected_candidate(tmp_path,monkeypatch):
    model=FakeModel(); original=model.call; selected=0; verify_count=0
    def call(task,data,trace):
        nonlocal selected,verify_count
        if task=='select':
            selected+=1
            items=original(task,data,trace)['evidence']
            if selected==1:
                # A misleading bridge initially claims both slots; actual deadline was rejected.
                a=next(p['id'] for p in data['candidates'] if '审批' in str(p))
                return {'evidence':[{'passage_id':a,'slot_id':s,'sentence_ids':[1]} for s in ('s1','s2')]}
            assert [s['id'] for s in data['slots']]==['s2']
            return {'evidence':[e for e in items if e['slot_id']=='s2']}
        if task=='verify':
            verify_count+=1
            if verify_count==1:
                return {'ok':False,'issues':['期限缺证'], 'gaps':[{'slot_id':'s2','query':'办理期限 三个工作日'}]}
            return {'ok':True,'issues':[], 'gaps':[]}
        return original(task,data,trace)
    model.call=call
    queries=[]
    def collect(self,qs,entities,config,trace):
        queries.append(qs); trace.add('fusion',ids=['a','b']); return ['a','b']
    monkeypatch.setattr(Retriever,'collect',collect)
    result=make(tmp_path,model).query('谁负责审批，期限多久？')
    repairs=[e for e in result.trace.events if e['stage']=='repair']
    assert repairs and repairs[0]['reason']=='verification_gap' and repairs[0]['added']==[]
    assert 'b' in repairs[0]['reconsidered'] and queries[1][0][0]=='s2'
    assert '期限' in result.answer and result.status=='complete' and not result.missing_slots
    assert len(repairs)<=2


def test_unknown_citation_is_not_fuzzily_repaired(tmp_path):
    model=FakeModel(); original=model.call
    def call(task,data,trace):
        if task=='answer': return {'answer':'张某负责审批。[p999]'}
        if task=='verify': return {'ok':True,'issues':[], 'gaps':[]}
        return original(task,data,trace)
    model.call=call
    result=make(tmp_path,model).query('谁负责审批？')
    assert result.status=='partial' and '[p999]' in result.answer
    assert model.calls['select']>0


def test_retrieval_budget_shared_with_verifier_and_no_infinite_loop(tmp_path):
    model=FakeModel(); original=model.call
    def call(task,data,trace):
        if task=='verify': return {'ok':False,'issues':['缺证'], 'gaps':[{'slot_id':'s2','query':'期限缺证'}]}
        return original(task,data,trace)
    model.call=call
    p=make(tmp_path,model,config=ChineseConfig(max_repairs=1))
    result=p.query('谁负责审批，期限多久？')
    assert len([e for e in result.trace.events if e['stage']=='repair'])==1
    assert result.status=='partial' and 's2' in result.missing_slots
    assert model.calls['answer']<=3  # initial + one retrieval repair + one answer rewrite


@pytest.mark.parametrize('answer', [
    '询问包括：1. 是否强迫购物；2. 是否兜售产品；3. 是否变更行程。[p1]',
    '询问包括：1、是否强迫购物；2、是否兜售产品；3、是否变更行程。[p1]',
    '询问包括：（1）是否强迫购物；（2）是否兜售产品；（3）是否变更行程。[p1]',
    '1. 是否强迫购物；2. 是否兜售产品；3. 是否变更行程。[p1]',
])
def test_inline_enumerations_are_format_not_numeric_facts(tmp_path, answer):
    p=make(tmp_path)
    p.model.call=lambda *args:{'ok':True,'issues':[], 'gaps':[]}
    payload={'passages':[{'id':'p1','text':'询问是否强迫购物、是否兜售产品、是否变更行程。'}]}
    assert p._verify(answer,payload,Trace())==[]


def test_inline_enumeration_does_not_remove_fact_values(tmp_path):
    p=make(tmp_path)
    p.model.call=lambda *args:{'ok':True,'issues':[], 'gaps':[]}
    payload={'passages':[{'id':'p1','text':'预算1.5万元，期限30天。'}]}
    issues=p._verify('包括：1. 预算2.5万元；2. 期限31天。[p1]',payload,Trace())
    assert issues==['numeric_copy_mismatch:2.5,31']


def test_equivalent_decimal_format_is_not_a_copy_error(tmp_path):
    p=make(tmp_path)
    p.model.call=lambda *args:{'ok':True,'issues':[], 'gaps':[]}
    payload={'passages':[{'id':'p1','text':'预算1,200.00万元，增长5.0%。'}]}
    assert p._verify('预算1200万元，增长5%。[p1]',payload,Trace())==[]
    assert p._verify('预算120万元，增长50%。[p1]',payload,Trace())==['numeric_copy_mismatch:120,50%']

@pytest.mark.parametrize('answer', [
    '1. 张某审批；2. 李某登记。[p1]',
    '原文p1列出：1. 张某审批；2. 李某登记。[p1]',
    '事项为（1）张某审批（2）李某登记。[p1]',
    '1）张某审批 2）李某登记。[p1]',
    '### 1. 张某审批。[p1]',
    '原文p1说明张某审批，问点s1已回答。[p1]',
])
def test_two_item_lists_and_prose_aliases_are_not_fact_numbers(tmp_path, answer):
    p=make(tmp_path)
    p.model.call=lambda *args:{'ok':True,'issues':[], 'gaps':[]}
    payload={'slots':[{'id':'s1','question':'谁审批？'}],
             'passages':[{'id':'p1','text':'张某审批，李某登记。'}]}
    assert p._verify(answer,payload,Trace())==[]


def test_failed_post_gap_generation_retains_matching_previous_answer(tmp_path,monkeypatch):
    model=FakeModel(); original=model.call; selected=0; answered=0
    def call(task,data,trace):
        nonlocal selected,answered
        if task=='select':
            selected+=1
            items=original(task,data,trace)['evidence']
            if selected==1:
                a=next(p['id'] for p in data['candidates'] if '审批' in str(p))
                return {'evidence':[{'passage_id':a,'slot_id':s,'sentence_ids':[1]} for s in ('s1','s2')]}
            return {'evidence':items}
        if task=='answer':
            answered+=1
            if answered>1:raise TimeoutError('regeneration failed')
            return original(task,data,trace)
        if task=='verify':
            return {'ok':False,'issues':['期限缺证'], 'gaps':[{'slot_id':'s2','query':'办理期限'}]}
        return original(task,data,trace)
    model.call=call
    def collect(self,qs,entities,config,trace):
        trace.add('fusion',ids=['a','b']);return ['a','b']
    monkeypatch.setattr(Retriever,'collect',collect)
    result=make(tmp_path,model).query('谁负责审批，期限多久？')
    assert result.status=='partial' and '张某' in result.answer and 's2' in result.missing_slots
    assert [p.id for p in result.passages]==['a']
    assert set(re.findall(r'\[([^\]]+)\]',result.answer))=={'a'}
    assert any(e['stage']=='answer_fallback' for e in result.trace.events)
    assert any(e['stage']=='answer_attempt' and e['answer']==result.answer for e in result.trace.events)
    assert answered==2
