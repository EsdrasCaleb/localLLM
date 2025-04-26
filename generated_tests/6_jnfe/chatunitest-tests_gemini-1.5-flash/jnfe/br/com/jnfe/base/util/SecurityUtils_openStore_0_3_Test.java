package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.Resource;
import java.io.ByteArrayInputStream;
import java.io.IOException;
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
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_0_3_Test {

    @Mock
    private Resource mockResource;

    @Test
    void testOpenStoreSuccess() throws Exception {
        when(mockResource.getFilename()).thenReturn("test.jks");
        when(mockResource.getInputStream()).thenReturn(new ByteArrayInputStream("test".getBytes()));
        KeyStore keyStore = SecurityUtils.openStore("JKS", mockResource, "changeit".toCharArray());
        assertNotNull(keyStore, "KeyStore should not be null");
        verify(mockResource, times(1)).getInputStream();
        verify(mockResource, times(1)).getFilename();
    }

    @Test
    void testOpenStoreInvalidType() {
        when(mockResource.getFilename()).thenReturn("test.jks");
        try {
            when(mockResource.getInputStream()).thenReturn(new ByteArrayInputStream("test".getBytes()));
            SecurityUtils.openStore("INVALID_TYPE", mockResource, "changeit".toCharArray());
            fail("Expected KeyStoreException");
        } catch (KeyStoreException e) {
            // Expected
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }

    @Test
    void testOpenStoreInvalidResource() throws IOException {
        when(mockResource.getInputStream()).thenThrow(new IOException("Resource not found"));
        assertThrows(Exception.class, () -> SecurityUtils.openStore("JKS", mockResource, "changeit".toCharArray()));
        verify(mockResource, times(1)).getInputStream();
    }

    @Test
    void testOpenStoreWrongPassword() throws IOException, KeyStoreException, NoSuchAlgorithmException, CertificateException {
        when(mockResource.getFilename()).thenReturn("test.jks");
        when(mockResource.getInputStream()).thenReturn(new ByteArrayInputStream("test".getBytes()));
        assertThrows(KeyStoreException.class, () -> SecurityUtils.openStore("JKS", mockResource, "wrongpassword".toCharArray()));
        verify(mockResource, times(1)).getInputStream();
    }
}
