package br.com.jnfe.base.util;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.springframework.core.io.FileSystemResource;

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_0_0_Test {

    @Mock
    private Resource keyStoreResource;

    private static final String KEY_STORE_TYPE = "JKS";

    private static final char[] PASSPHRASE = "password".toCharArray();

    private static final byte[] KEY_STORE_DATA = "dummy keystore data".getBytes();

    @BeforeEach
    public void setUp() throws IOException {
        when(keyStoreResource.getInputStream()).thenReturn(new ByteArrayInputStream(KEY_STORE_DATA));
        when(keyStoreResource.getFilename()).thenReturn("testKeystore.jks");
    }

    @Test
    public void testOpenStore_IOException() throws Exception {
        when(keyStoreResource.getInputStream()).thenThrow(new IOException("IOException"));
        Exception exception = assertThrows(IOException.class, () -> {
            SecurityUtils.openStore(KEY_STORE_TYPE, keyStoreResource, PASSPHRASE);
        });
        assertEquals("IOException", exception.getMessage());
    }
}
