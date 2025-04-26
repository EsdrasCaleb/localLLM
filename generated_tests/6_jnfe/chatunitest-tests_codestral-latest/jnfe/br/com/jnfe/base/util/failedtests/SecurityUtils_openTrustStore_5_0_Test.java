package br.com.jnfe.base.util;

import java.io.File;
import java.lang.reflect.Method;
import java.security.KeyStore;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
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
public class SecurityUtils_openTrustStore_5_0_Test {

    @InjectMocks
    private SecurityUtils securityUtils;

    @BeforeEach
    public void setUp() {
        SecurityUtils.trustStoreName = "cacerts";
        SecurityUtils.trustStorePassword = "changeit";
    }

    @Test
    public void testOpenTrustStoreWithNullPath() throws Exception {
        try (MockedStatic<SecurityUtils> mockedSecurityUtils = Mockito.mockStatic(SecurityUtils.class)) {
            String expectedPath = System.getProperty("java.home") + File.separatorChar + "lib" + File.separatorChar + "security" + File.separatorChar + "cacerts";
            char[] passphrase = "changeit".toCharArray();
            KeyStore mockKeyStore = mock(KeyStore.class);
            mockedSecurityUtils.when(() -> SecurityUtils.openStore(expectedPath, passphrase)).thenReturn(mockKeyStore);
            KeyStore result = SecurityUtils.openTrustStore(null, passphrase);
            assertNotNull(result);
            assertSame(mockKeyStore, result);
            mockedSecurityUtils.verify(() -> SecurityUtils.openStore(expectedPath, passphrase));
        }
    }

    @Test
    public void testOpenTrustStoreWithCustomPath() throws Exception {
        try (MockedStatic<SecurityUtils> mockedSecurityUtils = Mockito.mockStatic(SecurityUtils.class)) {
            String customPath = "/custom/path/to/truststore";
            String expectedPath = customPath + File.separatorChar + "cacerts";
            char[] passphrase = "changeit".toCharArray();
            KeyStore mockKeyStore = mock(KeyStore.class);
            mockedSecurityUtils.when(() -> SecurityUtils.openStore(expectedPath, passphrase)).thenReturn(mockKeyStore);
            KeyStore result = SecurityUtils.openTrustStore(customPath, passphrase);
            assertNotNull(result);
            assertSame(mockKeyStore, result);
            mockedSecurityUtils.verify(() -> SecurityUtils.openStore(expectedPath, passphrase));
        }
    }

    @Test
    public void testOpenTrustStoreException() throws Exception {
        try (MockedStatic<SecurityUtils> mockedSecurityUtils = Mockito.mockStatic(SecurityUtils.class)) {
            String expectedPath = System.getProperty("java.home") + File.separatorChar + "lib" + File.separatorChar + "security" + File.separatorChar + "cacerts";
            char[] passphrase = "changeit".toCharArray();
            mockedSecurityUtils.when(() -> SecurityUtils.openStore(expectedPath, passphrase)).thenThrow(new Exception("Test Exception"));
            Exception exception = assertThrows(Exception.class, () -> {
                SecurityUtils.openTrustStore(null, passphrase);
            });
            assertEquals("Test Exception", exception.getMessage());
            mockedSecurityUtils.verify(() -> SecurityUtils.openStore(expectedPath, passphrase));
        }
    }

    @Test
    public void testOpenStorePrivateMethod() throws Exception {
        String path = System.getProperty("java.home") + File.separatorChar + "lib" + File.separatorChar + "security" + File.separatorChar + "cacerts";
        char[] passphrase = "changeit".toCharArray();
        Method method = SecurityUtils.class.getDeclaredMethod("openStore", String.class, char[].class);
        method.setAccessible(true);
        KeyStore keyStore = (KeyStore) method.invoke(null, path, passphrase);
        assertNotNull(keyStore);
    }
}
