package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.util.Optional;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.io.FileInputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

@ExtendWith(MockitoExtension.class)
@MockitoSettings
public class SecurityUtils_openStore_0_0_Test {

    @Mock
    private Resource keyStoreResource;

    @Mock
    private char[] passphrase;

    @InjectMocks
    private SecurityUtils securityUtils;

    @Test
    public void testOpenStore_GivenValidKeyStoreResource_WhenCalled_ThenKeyStoreIsOpened() throws Exception {
        // Arrange
        Path keyStorePath = Paths.get("path/to/cacerts");
        when(keyStoreResource.getInputStream()).thenReturn(Files.newInputStream(keyStorePath));
        when(passphrase).thenReturn("validPassphrase".toCharArray());
        // Act
        KeyStore keyStore = securityUtils.openStore("JKS", keyStoreResource, passphrase);
        // Assert
        assertNotNull(keyStore);
    }

    @Test
    public void testOpenStore_GivenInvalidKeyStoreResource_WhenCalled_ThenKeyStoreExceptionIsThrown() throws Exception {
        // Arrange
        when(keyStoreResource.getInputStream()).thenThrow(IOException.class);
        // Act and Assert
        assertThrows(IOException.class, () -> securityUtils.openStore("JKS", keyStoreResource, passphrase));
    }

    @Test
    public void testOpenStore_GivenInvalidPassphrase_WhenCalled_ThenKeyStoreExceptionIsThrown() throws Exception {
        // Arrange
        when(keyStoreResource.getInputStream()).thenReturn(Files.newInputStream(Paths.get("path/to/cacerts")));
        when(passphrase).thenReturn("invalidPassphrase".toCharArray());
        // Act and Assert
        assertThrows(KeyStoreException.class, () -> securityUtils.openStore("JKS", keyStoreResource, passphrase));
    }

    @Test
    public void testOpenStore_GivenInvalidKeyStoreType_WhenCalled_ThenKeyStoreExceptionIsThrown() throws Exception {
        // Arrange
        when(keyStoreResource.getInputStream()).thenReturn(Files.newInputStream(Paths.get("path/to/cacerts")));
        when(passphrase).thenReturn("validPassphrase".toCharArray());
        // Act and Assert
        assertThrows(KeyStoreException.class, () -> securityUtils.openStore("InvalidKeyStoreType", keyStoreResource, passphrase));
    }
}
