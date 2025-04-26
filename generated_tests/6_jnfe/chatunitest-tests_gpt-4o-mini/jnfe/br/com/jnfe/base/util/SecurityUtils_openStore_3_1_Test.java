package br.com.jnfe.base.util;

import org.springframework.core.io.FileSystemResource;
import java.io.File;
import java.io.IOException;
import java.security.KeyStore;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

class SecurityUtils_openStore_3_1_Test {

    @Test
    void testOpenStore_InvalidStoreLocation_ThrowsIOException() {
        // Arrange
        String storeLocation = "invalidKeystore.jks";
        char[] passphrase = "validPassword".toCharArray();
        // Act & Assert
        assertThrows(IOException.class, () -> SecurityUtils.openStore(storeLocation, passphrase));
    }

    @Test
    void testOpenStore_EmptyPassphrase_ThrowsException() {
        // Arrange
        String storeLocation = "validKeystore.jks";
        char[] passphrase = "".toCharArray();
        // Act & Assert
        assertThrows(Exception.class, () -> SecurityUtils.openStore(storeLocation, passphrase));
    }
}
