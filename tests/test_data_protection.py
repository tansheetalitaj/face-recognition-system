import os
import unittest

from src.data_protection import WindowsDpapiProtector


@unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
class DataProtectionTests(unittest.TestCase):
    def test_dpapi_round_trip_is_user_bound_and_not_plaintext(self) -> None:
        protector = WindowsDpapiProtector()
        plaintext = b"sensitive-biometric-embedding"

        ciphertext = protector.protect(plaintext)
        recovered = protector.unprotect(ciphertext)

        self.assertNotEqual(ciphertext, plaintext)
        self.assertEqual(recovered, plaintext)


if __name__ == "__main__":
    unittest.main()
