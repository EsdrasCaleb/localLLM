package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.IOException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

@ExtendWith(MockitoExtension.class)
public class SecurityUtils_openStore_1_1_Test {

    @Mock
    private Resource keyStoreResource;

    private char[] passphrase = { 'p', 'a', 's', 's', 'p', 'h', 'r', 'a', 's', 'e' };

    @BeforeEach
    public void setUp() throws IOException {
        when(keyStoreResource.getFile()).thenReturn(new File("path/to/keystore"));
    }

    @Test
    public void testOpenStore() throws Exception {
        KeyStore keyStore = SecurityUtils.openStore(keyStoreResource, passphrase);
        assertEquals("JKS", keyStore.getType());
    }
}
