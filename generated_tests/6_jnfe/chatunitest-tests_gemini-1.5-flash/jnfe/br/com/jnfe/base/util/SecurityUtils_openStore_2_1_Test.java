package br.com.jnfe.base.util;

import org.junit.jupiter.api.io.TempDir;
import org.springframework.core.io.FileSystemResource;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
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
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

public class SecurityUtils_openStore_2_1_Test {

    @TempDir
    Path tempDir;

    @Test
    void testOpenStore_Success() throws Exception {
        // Create a temporary keystore file
        File keyStoreFile = tempDir.resolve("mykeystore.jks").toFile();
        KeyStore keyStore = KeyStore.getInstance("JKS");
        keyStore.load(null, null);
        try (FileOutputStream fos = new FileOutputStream(keyStoreFile)) {
            keyStore.store(fos, "changeit".toCharArray());
        }
        KeyStore result = SecurityUtils.openStore("JKS", keyStoreFile.getAbsolutePath(), "changeit".toCharArray());
        assertNotNull(result);
        assertEquals("JKS", result.getType());
    }

    @Test
    void testOpenStore_FileNotFound() {
        assertThrows(IOException.class, () -> {
            SecurityUtils.openStore("JKS", "nonexistentfile.jks", "changeit".toCharArray());
        });
    }

    @Test
    void testOpenStore_IncorrectPassword() throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException {
        File keyStoreFile = tempDir.resolve("mykeystore.jks").toFile();
        KeyStore keyStore = KeyStore.getInstance("JKS");
        keyStore.load(null, null);
        try (FileOutputStream fos = new FileOutputStream(keyStoreFile)) {
            keyStore.store(fos, "password123".toCharArray());
        }
        assertThrows(Exception.class, () -> {
            SecurityUtils.openStore("JKS", keyStoreFile.getAbsolutePath(), "wrongpassword".toCharArray());
        });
    }

    @Test
    void testOpenStore_InvalidKeyStoreType() {
        assertThrows(KeyStoreException.class, () -> {
            SecurityUtils.openStore("InvalidType", tempDir.resolve("mykeystore.jks").toFile().getAbsolutePath(), "changeit".toCharArray());
        });
    }

    @Test
    void testOpenStore_FileSystemResource() throws Exception {
        // Create a temporary keystore file
        File keyStoreFile = tempDir.resolve("mykeystore.jks").toFile();
        KeyStore keyStore = KeyStore.getInstance("JKS");
        keyStore.load(null, null);
        try (FileOutputStream fos = new FileOutputStream(keyStoreFile)) {
            keyStore.store(fos, "changeit".toCharArray());
        }
        KeyStore result = SecurityUtils.openStore("JKS", new FileSystemResource(keyStoreFile), "changeit".toCharArray());
        assertNotNull(result);
        assertEquals("JKS", result.getType());
    }

    @Test
    void testOpenStore_FileSystemResource_FileNotFound() {
        assertThrows(IOException.class, () -> {
            SecurityUtils.openStore("JKS", new FileSystemResource("nonexistentfile.jks"), "changeit".toCharArray());
        });
    }
}
