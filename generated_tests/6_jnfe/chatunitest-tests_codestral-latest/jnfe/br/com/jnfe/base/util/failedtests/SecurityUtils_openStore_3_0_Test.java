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
    void testOpenStore() throws Exception {
        try (MockedStatic<KeyStore> mockedKeyStore = Mockito.mockStatic(KeyStore.class)) {
            KeyStore mockKeyStore = mock(KeyStore.class);
            mockedKeyStore.when(() -> KeyStore.getInstance(anyString())).thenReturn(mockKeyStore);
            FileSystemResource mockResource = mock(FileSystemResource.class);
            when(mockResource.getInputStream()).thenReturn(new FileInputStream(new File(storeLocation)));
            KeyStore result = SecurityUtils.openStore(storeLocation, passphrase);
            assertNotNull(result);
            assertSame(mockKeyStore, result);
            verify(mockKeyStore).load(any(FileInputStream.class), eq(passphrase));
        }
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

    @Test
    void testOpenStore_KeyStoreException() throws Exception {
        try (MockedStatic<KeyStore> mockedKeyStore = Mockito.mockStatic(KeyStore.class)) {
            mockedKeyStore.when(() -> KeyStore.getInstance(anyString())).thenThrow(new KeyStoreException("Test Exception"));
            assertThrows(KeyStoreException.class, () -> SecurityUtils.openStore(storeLocation, passphrase));
        }
    }

    @Test
    void testOpenStore_NoSuchAlgorithmException() throws Exception {
        try (MockedStatic<KeyStore> mockedKeyStore = Mockito.mockStatic(KeyStore.class)) {
            mockedKeyStore.when(() -> KeyStore.getInstance(anyString())).thenThrow(new NoSuchAlgorithmException("Test Exception"));
            assertThrows(NoSuchAlgorithmException.class, () -> SecurityUtils.openStore(storeLocation, passphrase));
        }
    }

    @Test
    void testOpenStore_CertificateException() throws Exception {
        try (MockedStatic<KeyStore> mockedKeyStore = Mockito.mockStatic(KeyStore.class)) {
            KeyStore mockKeyStore = mock(KeyStore.class);
            mockedKeyStore.when(() -> KeyStore.getInstance(anyString())).thenReturn(mockKeyStore);
            doThrow(new CertificateException("Test Exception")).when(mockKeyStore).load(any(FileInputStream.class), eq(passphrase));
            assertThrows(CertificateException.class, () -> SecurityUtils.openStore(storeLocation, passphrase));
        }
    }
}
