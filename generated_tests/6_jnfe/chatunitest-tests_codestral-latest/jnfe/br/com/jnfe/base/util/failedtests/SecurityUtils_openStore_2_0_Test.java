package br.com.jnfe.base.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.FileSystemResource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_2_0_Test {

    @InjectMocks
    private SecurityUtils securityUtils;

    @Mock
    private FileSystemResource fileSystemResource;

    @BeforeEach
    public void setUp() {
        // Initialize mocks if needed
    }

    @Test
    public void testOpenStoreValid() throws Exception {
        String keyStoreType = "JKS";
        String storeLocation = "valid/path/to/keystore";
        char[] passphrase = "password".toCharArray();
        File file = new File(storeLocation);
        when(fileSystemResource.getFile()).thenReturn(file);
        KeyStore keyStore = SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        assertNotNull(keyStore);
        assertEquals(keyStoreType, keyStore.getType());
    }

    @Test
    public void testOpenStoreInvalidPath() {
        String keyStoreType = "JKS";
        String storeLocation = "invalid/path/to/keystore";
        char[] passphrase = "password".toCharArray();
        assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
    }

    @Test
    public void testOpenStoreInvalidPassphrase() {
        String keyStoreType = "JKS";
        String storeLocation = "valid/path/to/keystore";
        char[] passphrase = "wrongpassword".toCharArray();
        assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
    }

    @Test
    public void testOpenStoreInvalidKeyStoreType() {
        String keyStoreType = "INVALID";
        String storeLocation = "valid/path/to/keystore";
        char[] passphrase = "password".toCharArray();
        assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
    }
}
