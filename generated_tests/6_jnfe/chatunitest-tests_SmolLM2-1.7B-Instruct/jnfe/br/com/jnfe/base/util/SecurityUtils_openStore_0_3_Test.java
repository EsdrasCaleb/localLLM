package br.com.jnfe.base.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

public class SecurityUtils_openStore_0_3_Test {

    @Test
    public void testOpenStore_ValidInputs() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        Resource keyStoreResource = mock(Resource.class);
        char[] passphrase = "changeit".toCharArray();
        // Act
        KeyStore keyStore = SecurityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        // Assert
        assertNotNull(keyStore);
    }

    @Test
    public void testOpenStore_InvalidInputs() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        Resource keyStoreResource = mock(Resource.class);
        char[] passphrase = "changeit".toCharArray();
        // Act and Assert
        assertThrows(Exception.class, () -> SecurityUtils.openStore(keyStoreType, keyStoreResource, passphrase));
    }
}
