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

public class TransportKeyStoreBean_toString_7_0_Test {

    private TransportKeyStoreBean transportKeyStoreBean;

    @BeforeEach
    public void setUp() {
        transportKeyStoreBean = new TransportKeyStoreBean();
    }

    @Test
    public void testToString() {
        // Set system properties for testing
        System.setProperty("javax.net.ssl.keyStore", "testKeystore.jks");
        System.setProperty("javax.net.ssl.keyStoreType", "JKS");
        System.setProperty("javax.net.ssl.trustStore", "testTruststore.jks");
        System.setProperty("javax.net.ssl.trustStoreType", "JKS");
        String result = transportKeyStoreBean.toString();
        // Check if the output contains expected values
        assertTrue(result.contains("javax.net.ssl.keyStore='testKeystore.jks'"));
        assertTrue(result.contains("javax.net.ssl.keyStoreType='JKS'"));
        assertTrue(result.contains("javax.net.ssl.trustStoreType='JKS'"));
        assertTrue(result.contains("javax.net.ssl.trustStore='testTruststore.jks'"));
    }
}
