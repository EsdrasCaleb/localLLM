package br.com.jnfe.base.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import org.springframework.core.io.FileSystemResource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

class SecurityUtils_openStore_3_0_Test {

    private String storeLocation;

    private char[] passphrase;

    @BeforeEach
    void setUp() {
        storeLocation = "testKeystore.jks";
        passphrase = "testPassword".toCharArray();
    }

    @Test
    void testOpenStore_InvalidLocation() {
        String invalidLocation = "invalidKeystore.jks";
        assertThrows(IOException.class, () -> SecurityUtils.openStore(invalidLocation, passphrase));
    }

    @Test
    void testOpenStore_InvalidPassphrase() {
        char[] invalidPassphrase = "invalidPassword".toCharArray();
        assertThrows(IOException.class, () -> SecurityUtils.openStore(storeLocation, invalidPassphrase));
    }
}
