import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add scripts directory to sys.path to allow importing animatediff_mm
# This assumes the tests are run from the root of the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from animatediff_mm import AnimateDiffMM, MotionModuleType
# Note: May need to adjust motion_module import if MotionWrapper is directly used/mocked from its original location.
# from motion_module import MotionWrapper

# Mock global 'shared' and 'devices' objects from 'modules' that are used in animatediff_mm
# These would typically be part of the A1111/Forge environment
mock_shared = MagicMock()
mock_shared.cmd_opts = MagicMock()
mock_shared.cmd_opts.no_half = False
mock_shared.opts = MagicMock()
mock_shared.opts.data = {} # For animatediff_model_path

mock_devices = MagicMock()
mock_devices.device = 'cpu' # Mock device
mock_devices.cpu = 'cpu'
mock_devices.fp8 = False

# Patch 'modules.shared' and 'modules.devices' at the script level where animatediff_mm can access them
# sys.modules is used here to ensure the mocks are in place before animatediff_mm is potentially fully parsed.
sys.modules['modules.shared'] = mock_shared
sys.modules['modules.devices'] = mock_devices
sys.modules['modules.hashes'] = MagicMock()
sys.modules['modules.sd_models'] = MagicMock()
sys.modules['ldm.modules.diffusionmodules.util'] = MagicMock() # For GroupNorm32
sys.modules['sgm.modules.diffusionmodules.util'] = MagicMock() # For GroupNorm32 (SDXL)


class TestAnimateDiffMM(unittest.TestCase):

    def setUp(self):
        self.mm_instance = AnimateDiffMM()
        # Reset class variable for injection state between tests
        AnimateDiffMM.mm_injected = False
        # Set script_dir, normally done externally
        self.mm_instance.set_script_dir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

        # Mock MotionWrapper class that AnimateDiffMM instantiates
        self.mock_mw_instance = MagicMock()
        self.mock_mw_instance.mm_name = "initial_name"
        self.mock_mw_instance.load_state_dict = MagicMock()
        self.mock_mw_instance.to = MagicMock(return_value=self.mock_mw_instance)
        self.mock_mw_instance.eval = MagicMock()
        self.mock_mw_instance.half = MagicMock()
        self.mock_mw_instance.modules = MagicMock(return_value=[])
        self.mock_mw_instance.is_v2 = False
        self.mock_mw_instance.is_xl = False
        self.mock_mw_instance.enable_gn_hack = MagicMock(return_value=False)
        # Add other necessary attributes for MotionWrapper mock

    @patch('scripts.animatediff_mm.MotionWrapper') # Patch where it's looked up by AnimateDiffMM
    @patch('scripts.animatediff_mm.os.path.isfile', return_value=True)
    @patch('modules.sd_models.read_state_dict') # Patched at sys.modules level, direct use here
    @patch('modules.hashes.sha256')
    @patch('scripts.animatediff_mm.MotionModuleType.get_mm_type')
    def test_load_model_success(self, mock_get_mm_type, mock_sha256, mock_read_state_dict, mock_isfile, MockMotionWrapper):
        # Configure the mock MotionWrapper that is returned when MotionWrapper() is called
        MockMotionWrapper.return_value = self.mock_mw_instance

        mock_read_state_dict.return_value = {"test_key": "test_value"}
        mock_sha256.return_value = "test_hash"
        mock_get_mm_type.return_value = MotionModuleType.STANDARD

        # Action
        self.mm_instance.load("test_model.safetensors")

        # Assertions
        self.assertIsNotNone(self.mm_instance.mm)
        MockMotionWrapper.assert_called_once_with(mm_name="test_model.safetensors", mm_hash="test_hash", mm_type=MotionModuleType.STANDARD)
        self.mm_instance.mm.load_state_dict.assert_called_once_with({"test_key": "test_value"})
        self.mm_instance.mm.to.assert_called_with('cpu') # from mock_devices.device
        self.mm_instance.mm.eval.assert_called_once()
        # self.mm_instance.mm.half.assert_called_once() # Depends on no_half

    @patch('scripts.animatediff_mm.os.path.isfile', return_value=False)
    def test_load_model_file_not_found(self, mock_isfile):
        with self.assertRaises(RuntimeError) as context:
            self.mm_instance.load("non_existent_model.safetensors")
        self.assertIn("Please download models manually.", str(context.exception))

if __name__ == '__main__':
    # Create a 'tests' directory if it doesn't exist
    # This check is more for running the script directly; create_file_with_block handles dir creation.
    if not os.path.exists(os.path.join(os.path.dirname(__file__))):
         os.makedirs(os.path.join(os.path.dirname(__file__)))
    unittest.main()
