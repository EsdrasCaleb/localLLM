package br.com.jnfe.base.util;

import static org.mockito.ArgumentMatchers.any;
import org.springframework.core.io.FileSystemResource;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
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

public class SecurityUtils_openStore_2_4_Test {

    @Mock
    private FileSystemResource fileSystemResource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testOpenStore_Success() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        String storeLocation = "path/to/keystore.jks";
        char[] passphrase = "password".toCharArray();
        KeyStore keyStore = KeyStore.getInstance(keyStoreType);
        FileInputStream fileInputStream = mock(FileInputStream.class);
        when(fileSystemResource.getInputStream()).thenReturn(fileInputStream);
        doNothing().when(keyStore).load(fileInputStream, passphrase);
        // Act
        KeyStore result = SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        // Assert
        assertEquals(keyStore, result);
    }

    @Test
    public void testOpenStore_KeyStoreException() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        String storeLocation = "path/to/keystore.jks";
        char[] passphrase = "password".toCharArray();
        KeyStore keyStore = KeyStore.getInstance(keyStoreType);
        FileInputStream fileInputStream = mock(FileInputStream.class);
        when(fileSystemResource.getInputStream()).thenReturn(fileInputStream);
        doThrow(new KeyStoreException()).when(keyStore).load(fileInputStream, passphrase);
        // Act & Assert
        Exception exception = assertThrows(KeyStoreException.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
        assertNotNull(exception);
    }

    @Test
    public void testOpenStore_IOException() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        String storeLocation = "path/to/keystore.jks";
        char[] passphrase = "password".toCharArray();
        when(fileSystemResource.getInputStream()).thenThrow(new IOException());
        // Act & Assert
        Exception exception = assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
        assertNotNull(exception);
    }

    @Test
    public void testOpenStore_NoSuchAlgorithmException() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        String storeLocation = "path/to/keystore.jks";
        char[] passphrase = "password".toCharArray();
        when(KeyStore.getInstance(keyStoreType)).thenThrow(new NoSuchAlgorithmException());
        // Act & Assert
        Exception exception = assertThrows(NoSuchAlgorithmException.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
        assertNotNull(exception);
    }

    @Test
    public void testOpenStore_CertificateException() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        String storeLocation = "path/to/keystore.jks";
        char[] passphrase = "password".toCharArray();
        KeyStore keyStore = KeyStore.getInstance(keyStoreType);
        FileInputStream fileInputStream = mock(FileInputStream.class);
        when(fileSystemResource.getInputStream()).thenReturn(fileInputStream);
        doThrow(new CertificateException()).when(keyStore).load(fileInputStream, passphrase);
        // Act & Assert
        Exception exception = assertThrows(CertificateException.class, () -> {
            SecurityUtils.openStore(keyStoreType, storeLocation, passphrase);
        });
        assertNotNull(exception);
    }
}
