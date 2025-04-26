package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import java.util.Arrays;
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
public class SecurityUtils_openStore_0_1_Test {

    @Mock
    private InputStream inputStream;

    @Mock
    private Resource keyStoreResource;

    @InjectMocks
    private SecurityUtils securityUtils;

    @Test
    public void testOpenStore_Success() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        char[] passphrase = "changeit".toCharArray();
        byte[] keyStoreData = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        InputStream mockInputStream = new ByteArrayInputStream(keyStoreData);
        when(keyStoreResource.getInputStream()).thenReturn(mockInputStream);
        KeyStore expectedKeyStore = KeyStore.getInstance(keyStoreType);
        when(KeyStore.getInstance(keyStoreType)).thenReturn(expectedKeyStore);
        when(inputStream.available()).thenReturn(keyStoreData.length);
        // Crucial:  Use a Mockito answer to simulate reading from the stream
        when(inputStream.read(any(byte[].class), anyInt(), anyInt())).thenAnswer(invocation -> {
            byte[] buffer = invocation.getArgument(0);
            int offset = invocation.getArgument(1);
            int len = invocation.getArgument(2);
            int bytesRead = Math.min(len, keyStoreData.length - offset);
            System.arraycopy(keyStoreData, offset, buffer, 0, bytesRead);
            return bytesRead;
        });
        expectedKeyStore.load(mockInputStream, passphrase);
        // Act
        KeyStore actualKeyStore = securityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        // Assert
        assertEquals(expectedKeyStore, actualKeyStore);
        // Crucial: Verify that the input stream is closed.
        verify(mockInputStream).close();
    }

    @Test
    public void testOpenStore_Exception() throws IOException {
        // Arrange
        String keyStoreType = "JKS";
        char[] passphrase = "changeit".toCharArray();
        when(keyStoreResource.getInputStream()).thenThrow(new IOException("Failed to get input stream"));
        // Act & Assert (expecting exception)
        assertThrows(IOException.class, () -> {
            securityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        });
    }

    @Test
    public void testOpenStore_KeyStoreException() throws Exception {
        // Arrange
        String keyStoreType = "JKS";
        char[] passphrase = "changeit".toCharArray();
        when(keyStoreResource.getInputStream()).thenReturn(inputStream);
        when(KeyStore.getInstance(keyStoreType)).thenThrow(new KeyStoreException("Failed to create KeyStore"));
        // Act & Assert (expecting exception)
        assertThrows(KeyStoreException.class, () -> {
            securityUtils.openStore(keyStoreType, keyStoreResource, passphrase);
        });
    }
}
