package br.com.jnfe.base;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.net.ssl.KeyManagerFactory;
import java.security.*;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

@ExtendWith(MockitoExtension.class)
public class TransportKeyStoreBean_openTransportKeyManagerFactory_9_0_Test {

    @Mock
    private KeyStore keyStore;

    @InjectMocks
    private TransportKeyStoreBean transportKeyStoreBean;

    @Test
    void testOpenTransportKeyManagerFactory_Success() throws Exception {
        transportKeyStoreBean.setKeyStorePassword("password");
        PrivateKey mockPrivateKey = mock(PrivateKey.class);
        when(keyStore.getKey("keyAlias", "password".toCharArray())).thenReturn(mockPrivateKey);
        X509Certificate mockCertificate = mock(X509Certificate.class);
        Enumeration<String> aliases = Collections.enumeration(Arrays.asList("keyAlias"));
        when(keyStore.aliases()).thenReturn(aliases);
        when(keyStore.getCertificate("keyAlias")).thenReturn(mockCertificate);
        KeyManagerFactory kmf = transportKeyStoreBean.openTransportKeyManagerFactory();
        assertNotNull(kmf);
        verify(keyStore).getKey("keyAlias", "password".toCharArray());
    }

    @Test
    void testOpenTransportKeyManagerFactory_NullPassword() {
        assertThrows(NullPointerException.class, () -> transportKeyStoreBean.openTransportKeyManagerFactory());
    }

    @Test
    void testOpenTransportKeyManagerFactory_KeyStoreException() throws Exception {
        transportKeyStoreBean.setKeyStorePassword("password");
        doThrow(new KeyStoreException("Keystore exception")).when(keyStore).load(any(), any());
        assertThrows(Exception.class, () -> transportKeyStoreBean.openTransportKeyManagerFactory());
    }

    @Test
    void testOpenTransportKeyManagerFactory_NoSuchAlgorithmException() throws Exception {
        transportKeyStoreBean.setKeyStorePassword("password");
        doThrow(new NoSuchAlgorithmException("Algorithm not found")).when(keyStore).load(any(), any());
        assertThrows(Exception.class, () -> transportKeyStoreBean.openTransportKeyManagerFactory());
    }

    @Test
    void testOpenTransportKeyManagerFactory_CertificateException() throws Exception {
        transportKeyStoreBean.setKeyStorePassword("password");
        doThrow(new CertificateException("Certificate exception")).when(keyStore).load(any(), any());
        assertThrows(Exception.class, () -> transportKeyStoreBean.openTransportKeyManagerFactory());
    }

    @Test
    void testOpenTransportKeyManagerFactory_IOException() throws Exception {
        transportKeyStoreBean.setKeyStorePassword("password");
        doThrow(new java.io.IOException("IO Exception")).when(keyStore).load(any(), any());
        assertThrows(Exception.class, () -> transportKeyStoreBean.openTransportKeyManagerFactory());
    }
}
