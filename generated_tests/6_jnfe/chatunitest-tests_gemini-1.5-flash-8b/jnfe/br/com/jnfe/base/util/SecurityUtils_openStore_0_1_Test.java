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
}
