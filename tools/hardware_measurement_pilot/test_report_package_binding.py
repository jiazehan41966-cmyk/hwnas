"""Package identity selection tests; static hardware acceptance is a later gate."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.hardware_measurement_pilot.report import _bound_package_case


class PackageBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.attempt = Path(self.temporary.name)
        self.case = dict(case_id='example', case_sha256='a'*64)
        self.source = 'b'*64
        self.package = self.attempt/'board_packages/example/old_fetch/case'
        self.package.mkdir(parents=True)
        self.bit_sha = hashlib.sha256(b'bit bytes').hexdigest()
        self.route = dict(**self.case, source_manifest_sha256=self.source, status='PASS',
                          gate=dict(artifacts=[dict(path='vivado_shell/top.bit',sha256=self.bit_sha)]))
        payloads = {'case.json':json.dumps(self.case).encode(),
                    'evidence/route_status.json':json.dumps(self.route).encode(),
                    'vivado_shell/top.bit':b'bit bytes','vivado_shell/top.hwh':b'bound HWH bytes'}
        for name,payload in payloads.items():
            path = self.package/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(payload)
        self.transport = dict(**self.case,source_manifest_sha256=self.source,
                              files={name:hashlib.sha256(payload).hexdigest() for name,payload in payloads.items()})
        self.write_transport()
        self.remote = self.attempt/'remote_evidence/new_fetch/results/example/route/status.json'
        self.remote.parent.mkdir(parents=True)
        self.remote.write_text(json.dumps(self.route))

    def write_transport(self):
        (self.package/'package_transport_manifest.json').write_text(json.dumps(self.transport))

    def resolve(self,route=None,source=None):
        return _bound_package_case(self.attempt,self.case,(self.remote,self.route if route is None else route),
                                   self.bit_sha,self.source if source is None else source)

    def test_new_remote_record_finds_same_bound_old_local_package(self):
        package,evidence=self.resolve()
        self.assertEqual(package,self.package)
        self.assertEqual(evidence['status'],'PASS_CASE_BIT_SOURCE_PACKAGE_BINDING')
        self.assertEqual(evidence['selected_route_status_source'],self.remote.as_posix())

    def test_selected_route_case_and_source_mismatch_are_rejected(self):
        for field,value in [('case_sha256','c'*64),('source_manifest_sha256','c'*64),('status','FAILED')]:
            with self.subTest(field=field):
                self.assertIsNone(self.resolve(route=dict(self.route,**{field:value}))[0])
        self.assertIsNone(self.resolve(source='c'*64)[0])

    def test_package_case_source_or_bit_identity_cannot_be_substituted(self):
        original=json.loads(json.dumps(self.transport))
        for field in ('case_sha256','source_manifest_sha256'):
            with self.subTest(field=field):
                self.transport=dict(original,**{field:'c'*64})
                self.write_transport()
                self.assertIsNone(self.resolve()[0])
        self.transport=json.loads(json.dumps(original))
        self.transport['files']['vivado_shell/top.bit']='c'*64
        self.write_transport()
        self.assertIsNone(self.resolve()[0])
        self.transport=original
        self.write_transport()
        (self.package/'case.json').write_text(json.dumps(dict(self.case,case_sha256='c'*64)))
        self.assertIsNone(self.resolve()[0])

    def test_changed_bound_bit_or_hwh_bytes_cannot_be_accepted(self):
        for name in ('vivado_shell/top.bit','vivado_shell/top.hwh'):
            with self.subTest(name=name):
                path=self.package/name
                original=path.read_bytes()
                path.write_bytes(original+b'tampered')
                self.assertIsNone(self.resolve()[0])
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
