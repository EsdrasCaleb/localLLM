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
    void testOpenStoreWithNullResource() {
        assertThrows(NullPointerException.class, () -> {
            SecurityUtils.openStore((Resource) null, passphrase);
        });
    }
}
