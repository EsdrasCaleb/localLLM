package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyStore;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileInputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_0_0_Test {

    private static final String KEYSTORE_TYPE = "JKS";

    private static final char[] PASSPHRASE = { 'p', 'a', 's', 's', 'p', 'h', 'r', 'a', 's', 'e' };

    @Mock
    private Resource keyStoreResource;

    @Mock
    private Logger logger;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testOpenStore() throws Exception {
        // Arrange
        when(keyStoreResource.getFilename()).thenReturn("cacerts");
        when(keyStoreResource.getInputStream()).thenReturn(getClass().getResourceAsStream("/cacerts"));
        // Act
        KeyStore keyStore = SecurityUtils.openStore(KEYSTORE_TYPE, keyStoreResource, PASSPHRASE);
        // Assert
        assertEquals("JKS", keyStore.getType());
    }
}
