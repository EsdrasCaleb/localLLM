package br.com.jnfe.base.util;

import org.springframework.core.io.Resource;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.security.KeyStore;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileInputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;

public class SecurityUtils_openStore_0_1_Test {

    @Test
    public void testOpenStore_InvalidKeyStoreType_ShouldThrowException() {
        // Arrange
        String keyStoreType = "INVALID_TYPE";
        char[] passphrase = "changeit".toCharArray();
        Resource keyStoreResource = Mockito.mock(Resource.class);
        // Act & Assert
        Assertions.assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        });
    }

    @Test
    public void testOpenStore_NullResource_ShouldThrowException() {
        // Arrange
        String keyStoreType = "JKS";
        char[] passphrase = "changeit".toCharArray();
        Resource keyStoreResource = null;
        // Act & Assert
        Assertions.assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        });
    }

    @Test
    public void testOpenStore_ResourceThrowsException_ShouldThrowException() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        char[] passphrase = "changeit".toCharArray();
        Resource keyStoreResource = Mockito.mock(Resource.class);
        Mockito.when(keyStoreResource.getInputStream()).thenThrow(new IOException("Resource not found"));
        // Act & Assert
        Assertions.assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        });
    }
}
