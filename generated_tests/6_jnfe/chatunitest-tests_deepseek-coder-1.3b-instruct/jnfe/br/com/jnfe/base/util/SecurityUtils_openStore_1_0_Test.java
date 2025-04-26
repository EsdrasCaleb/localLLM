package br.com.jnfe.base.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

@ExtendWith(MockitoExtension.class)
class SecurityUtils_openStore_1_0_Test {

    @InjectMocks
    SecurityUtils securityUtils;

    @Test
    void openStoreTest() throws Exception {
        // given
        Resource mockResource = mock(Resource.class);
        char[] passphrase = { 'p', 'a', 's', 's', 'w', 'o', 'r', 'd' };
        // when
        KeyStore result = securityUtils.openStore(mockResource, passphrase);
        // then
        verify(mockResource, times(1)).getInputStream();
        assertNotNull(result);
    }
}
