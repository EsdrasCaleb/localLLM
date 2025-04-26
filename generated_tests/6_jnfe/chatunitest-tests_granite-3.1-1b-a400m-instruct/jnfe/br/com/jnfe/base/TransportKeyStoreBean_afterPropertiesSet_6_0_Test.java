package br.com.jnfe.base;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

public class TransportKeyStoreBean_afterPropertiesSet_6_0_Test {

    private TransportKeyStoreBean transportKeyStoreBean;

    @BeforeEach
    public void setUp() {
        transportKeyStoreBean = new TransportKeyStoreBean();
    }

    @Test
    public void testAfterPropertiesSet() throws Exception {
        // Mocking System.setProperty to set default values
        System.setProperty("javax.net.ssl.keyStoreType", "PKCS12");
        System.setProperty("javax.net.ssl.keyStore", "file:/path/to/keystore");
        System.setProperty("javax.net.ssl.keyStorePassword", "password123");
        System.setProperty("javax.net.ssl.trustStoreType", "JKS");
        System.setProperty("javax.net.ssl.trustStore", "file:/path/to/truststore");
        System.setProperty("javax.net.ssl.trustStorePassword", "trustpassword123");
        transportKeyStoreBean.afterPropertiesSet();
        // Verify properties are set correctly
        assertEquals(System.getProperty("javax.net.ssl.keyStoreType"), "PKCS12");
        assertEquals(System.getProperty("javax.net.ssl.keyStore"), "file:/path/to/keystore");
        assertEquals(System.getProperty("javax.net.ssl.keyStorePassword"), "password123");
        assertEquals(System.getProperty("javax.net.ssl.trustStoreType"), "JKS");
        assertEquals(System.getProperty("javax.net.ssl.trustStore"), "file:/path/to/truststore");
        assertEquals(System.getProperty("javax.net.ssl.trustStorePassword"), "trustpassword123");
    }
}
