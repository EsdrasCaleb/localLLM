package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.Resource;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
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

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_1_1_Test {

    @Mock
    private Resource keyStoreResource;

    @InjectMocks
    private SecurityUtils securityUtils;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testOpenStore_Success() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        KeyStore mockKeyStore = mock(KeyStore.class);
        InputStream inputStream = mock(InputStream.class);
        when(keyStoreResource.getInputStream()).thenReturn(inputStream);
        when(mockKeyStore.getType()).thenReturn(KeyStore.getDefaultType());
        when(KeyStore.getInstance(KeyStore.getDefaultType())).thenReturn(mockKeyStore);
        // Act
        KeyStore result = securityUtils.openStore(keyStoreResource, passphrase);
        // Assert
        assertNotNull(result);
        assertEquals(KeyStore.getDefaultType(), result.getType());
    }

    @Test
    public void testOpenStore_CertificateException() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        KeyStore mockKeyStore = mock(KeyStore.class);
        InputStream inputStream = mock(InputStream.class);
        when(keyStoreResource.getInputStream()).thenReturn(inputStream);
        CertificateException certificateException = new CertificateException("Failed to load certificate");
        when(KeyStore.getInstance(KeyStore.getDefaultType())).thenReturn(mockKeyStore);
        doThrow(certificateException).when(mockKeyStore).load(any(InputStream.class), any(char[].class));
        // Act & Assert
        Exception exception = assertThrows(Exception.class, () -> {
            securityUtils.openStore(keyStoreResource, passphrase);
        });
        // Assert
        assertEquals(certificateException.getMessage(), exception.getMessage());
    }

    @Test
    public void testOpenStore_KeyStoreException() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        InputStream inputStream = mock(InputStream.class);
        when(keyStoreResource.getInputStream()).thenReturn(inputStream);
        KeyStoreException keyStoreException = new KeyStoreException("Failed to get KeyStore instance");
        when(KeyStore.getInstance(KeyStore.getDefaultType())).thenThrow(keyStoreException);
        // Act & Assert
        Exception exception = assertThrows(Exception.class, () -> {
            securityUtils.openStore(keyStoreResource, passphrase);
        });
        // Assert
        assertEquals(keyStoreException.getMessage(), exception.getMessage());
    }

    @Test
    public void testOpenStore_NoSuchAlgorithmException() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        InputStream inputStream = mock(InputStream.class);
        when(keyStoreResource.getInputStream()).thenReturn(inputStream);
        NoSuchAlgorithmException noSuchAlgorithmException = new NoSuchAlgorithmException("Algorithm not found");
        when(KeyStore.getInstance(KeyStore.getDefaultType())).thenThrow(noSuchAlgorithmException);
        // Act & Assert
        Exception exception = assertThrows(Exception.class, () -> {
            securityUtils.openStore(keyStoreResource, passphrase);
        });
        // Assert
        assertEquals(noSuchAlgorithmException.getMessage(), exception.getMessage());
    }

    @Test
    public void testOpenStore_IOException() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        InputStream inputStream = mock(InputStream.class);
        when(keyStoreResource.getInputStream()).thenThrow(new IOException("Failed to get input stream"));
        // Act & Assert
        Exception exception = assertThrows(Exception.class, () -> {
            securityUtils.openStore(keyStoreResource, passphrase);
        });
        // Assert
        assertTrue(exception.getMessage().contains("Failed to get input stream"));
    }

    @Test
    public void testOpenStore_ResourceIOException() throws Exception {
        // Arrange
        char[] passphrase = "changeit".toCharArray();
        IOException ioException = new IOException("Failed to read resource");
        when(keyStoreResource.getInputStream()).thenThrow(ioException);
        // Act & Assert
        Exception exception = assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreResource, passphrase);
        });
        // Assert
        assertEquals(ioException.getMessage(), exception.getMessage());
    }
}
