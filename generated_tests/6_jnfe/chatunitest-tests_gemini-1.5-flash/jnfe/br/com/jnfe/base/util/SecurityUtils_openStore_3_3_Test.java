package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import javax.net.ssl.KeyStoreBuilderParameters;
import javax.net.ssl.TrustManagerFactory;
import java.io.File;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableKeyException;
import java.security.cert.CertificateException;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_3_3_Test {

    @Test
    void testOpenStoreValidPath() throws Exception {
        // Create a dummy KeyStore for testing
        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        // Initialize empty KeyStore
        keyStore.load(null, null);
        String tempFilePath = File.createTempFile("testKeyStore", ".jks").getAbsolutePath();
        keyStore.store(new java.io.FileOutputStream(tempFilePath), "changeit".toCharArray());
        KeyStore result = SecurityUtils.openStore(tempFilePath, "changeit".toCharArray());
        assertNotNull(result);
        assertEquals(keyStore.size(), result.size());
        new File(tempFilePath).delete();
    }

    @Test
    void testOpenStoreInvalidPath() {
        assertThrows(Exception.class, () -> SecurityUtils.openStore("invalid/path", "changeit".toCharArray()));
    }

    @Test
    void testOpenStoreInvalidPassword() {
        assertThrows(Exception.class, () -> SecurityUtils.openStore("cacerts", "wrongpassword".toCharArray()));
    }

    @Test
    void testOpenStoreNullPassword() {
        assertThrows(Exception.class, () -> SecurityUtils.openStore("cacerts", null));
    }

    @Test
    void testOpenStoreNullPath() {
        assertThrows(Exception.class, () -> SecurityUtils.openStore((String) null, "changeit".toCharArray()));
    }

    @Test
    void testOpenStoreEmptyPath() {
        assertThrows(Exception.class, () -> SecurityUtils.openStore("", "changeit".toCharArray()));
    }

    @Test
    void testOpenStoreResourceValid() throws Exception {
        String tempFilePath = File.createTempFile("testKeyStore", ".jks").getAbsolutePath();
        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        keyStore.load(null, null);
        keyStore.store(new java.io.FileOutputStream(tempFilePath), "changeit".toCharArray());
        Resource resource = new FileSystemResource(tempFilePath);
        KeyStore result = SecurityUtils.openStore(resource, "changeit".toCharArray());
        assertNotNull(result);
        assertEquals(keyStore.size(), result.size());
        new File(tempFilePath).delete();
    }

    @Test
    void testOpenStoreResourceInvalid() {
        Resource resource = new FileSystemResource("invalid/path");
        assertThrows(Exception.class, () -> SecurityUtils.openStore(resource, "changeit".toCharArray()));
    }

    @Test
    void testOpenStoreResourceNull() {
        assertThrows(Exception.class, () -> SecurityUtils.openStore((Resource) null, "changeit".toCharArray()));
    }
}
