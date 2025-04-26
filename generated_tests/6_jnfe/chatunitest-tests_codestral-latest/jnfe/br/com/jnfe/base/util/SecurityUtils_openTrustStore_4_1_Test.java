package br.com.jnfe.base.util;

import java.io.File;
import java.io.FileInputStream;
import java.security.KeyStore;
import org.slf4j.Logger;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

class SecurityUtils_openTrustStore_4_1_Test {

    @InjectMocks
    private SecurityUtils securityUtils;

    @Mock
    private Logger logger;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testOpenTrustStore() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        String trustStorePath = "src/test/resources/cacerts";
        // Act
        KeyStore keyStore = SecurityUtils.openTrustStore(passphrase);
        // Assert
        assertNotNull(keyStore);
        assertEquals("JKS", keyStore.getType());
    }

    @Test
    void testOpenTrustStoreWithInvalidPassphrase() {
        // Arrange
        char[] passphrase = "invalid".toCharArray();
        // Act & Assert
        assertThrows(Exception.class, () -> {
            SecurityUtils.openTrustStore(passphrase);
        });
    }

    @Test
    void testOpenTrustStoreWithNullPassphrase() {
        // Arrange
        char[] passphrase = null;
        // Act & Assert
        assertThrows(NullPointerException.class, () -> {
            SecurityUtils.openTrustStore(passphrase);
        });
    }
}
