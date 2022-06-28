import contextlib
import io
import unittest
from farm.software import wsclean, which
from farm.software.wsclean import _WSCLEAN_DEFAULT_ARGS, Option


class TestWsclean(unittest.TestCase):
    def test_wsclean(self):
        ms_list = ['test1.ms', 'test2.ms', 'test3.ms']
        input_args = {
            'join-polarizations': False,  # False is the default
            'auto-mask': None,
            'local-rms': True,
            'local-rms-window': 25,  # 25 is the default
            'local-rms-method': 'rms',  # rms is the default
            'gain': 0.05,  # 0.1 is the default
            'mgain': 0.5,  # 1.0 is the default
            'join-channels': True,
            'multiscale': True,
            'multiscale-scale-bias': 0.6,  # 0.6 is the default
            'multiscale-scales': '4,10,20,50',
            'multiscale-shape': 'gaussian',
            'name': 'unittest_wsclean',
            'size': f'{1024} {1024}',
            'scale': f'{6.0}asec',
        }

        # Use contextlib.redirect_stderr context manager to stop Error printing
        # out to console as it is raised in the code
        with contextlib.redirect_stderr(None):
            with self.assertRaises(KeyError) as context:
                non_wsclean_arg = {'not_a_wsclean_arg': 'not_a_wsclean_val'}
                wsclean(ms_list, kw_args=input_args | non_wsclean_arg,
                        dryrun=True)
        self.assertEqual(f'"\'{list(non_wsclean_arg.keys())[0]}\' not a valid '
                         f'wsclean command-line arg"', str(context.exception))

        cmd, products = wsclean(ms_list, kw_args=input_args, dryrun=True)

        self.assertTrue(cmd.startswith(str(which('wsclean'))))
        self.assertTrue(cmd.endswith(' '.join(ms_list)))
        self.assertEqual(['dirty', 'image', 'model', 'psf', 'residual'],
                         sorted(products.keys()))

        for k, v in input_args.items():
            if isinstance(_WSCLEAN_DEFAULT_ARGS[k], Option):
                arg = f'-{k} '
                if _WSCLEAN_DEFAULT_ARGS[k].value == v:
                    self.assertNotIn(arg, cmd)
                else:
                    self.assertIn(arg, cmd)
            else:
                arg = f'-{k} {v}'
                if _WSCLEAN_DEFAULT_ARGS[k] == v:
                    self.assertNotIn(arg, cmd)
                else:
                    self.assertIn(arg, cmd)


if __name__ == '__main__':
    unittest.main()
