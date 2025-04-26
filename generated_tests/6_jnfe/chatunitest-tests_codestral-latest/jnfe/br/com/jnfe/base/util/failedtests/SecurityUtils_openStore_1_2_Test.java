package br.com.jnfe.base.util;

import java.io.InputStream;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.Arrays;
import org.springframework.core.io.Resource;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class SecurityUtils_openStore_1_2_Test {

    private Resource mockResource;

    private char[] passphrase;

    @BeforeEach
    void setUp() {
        mockResource = mock(Resource.class);
        passphrase = "password".toCharArray();
    }

    @Test
    void testOpenStore() throws Exception {
        // Mock the InputStream from the Resource
        InputStream mockInputStream = mock(InputStream.class);
        when(mockResource.getInputStream()).thenReturn(mockInputStream);
        // Create a mock KeyStore
        KeyStore mockKeyStore = Mockito.mock(KeyStore.class);
        when(mockKeyStore.getType()).thenReturn(KeyStore.getDefaultType());
        // Mock the KeyStore loading process
        doNothing().when(mockKeyStore).load(mockInputStream, passphrase);
        // Mock the KeyStore.getInstance method
        when(KeyStore.getInstance(anyString())).thenReturn(mockKeyStore);
        // Invoke the method under test
        KeyStore result = SecurityUtils.openStore(mockResource, passphrase);
        // Verify the result
        assertNotNull(result);
        assertEquals(mockKeyStore, result);
        // Verify interactions
        verify(mockResource).getInputStream();
        verify(mockKeyStore).load(mockInputStream, passphrase);
    }

    @Test
    void testOpenStoreWithNullResource() {
        assertThrows(NullPointerException.class, () -> {
            SecurityUtils.openStore((Resource) null, passphrase);
        });
    }

    @Test
    void testOpenStoreWithNullPassphrase() {
        assertThrows(NullPointerException.class, () -> {
            SecurityUtils.openStore(mockResource, (char[]) null);
        });
    }

    @Test
    void testOpenStoreWithInvalidPassphrase() throws Exception {
        // Mock the InputStream from the Resource
        InputStream mockInputStream = mock(InputStream.class);
        when(mockResource.getInputStream()).thenReturn(mockInputStream);
        // Create a mock KeyStore
        KeyStore mockKeyStore = Mockito.mock(KeyStore.class);
        when(mockKeyStore.getType()).thenReturn(KeyStore.getDefaultType());
        // Mock the KeyStore loading process to throw an exception
        doThrow(new java.io.IOException("Invalid passphrase")).when(mockKeyStore).load(mockInputStream, passphrase);
        // Mock the KeyStore.getInstance method
        when(KeyStore.getInstance(anyString())).thenReturn(mockKeyStore);
        // Invoke the method under test and expect an exception
        assertThrows(Exception.class, () -> {
            SecurityUtils.openStore(mockResource, passphrase);
        });
        // Verify interactions
        verify(mockResource).getInputStream();
        verify(mockKeyStore).load(mockInputStream, passphrase);
    }
}
