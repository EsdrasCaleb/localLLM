package br.com.jnfe.base;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

class TransportKeyStoreBean_afterPropertiesSet_6_0_Test {

    @InjectMocks
    private TransportKeyStoreBean transportKeyStoreBean;

    @Mock
    private Logger logger;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        transportKeyStoreBean = new TransportKeyStoreBean();
        logger = LoggerFactory.getLogger(TransportKeyStoreBean.class);
    }

    @Test
    void testAfterPropertiesSetWithAllProperties() throws Exception {
        transportKeyStoreBean.setKeyStoreType("customType");
        transportKeyStoreBean.setKeyStoreUri("customUri");
        transportKeyStoreBean.setKeyStorePassword("customPassword");
        transportKeyStoreBean.setTrustStore("customTrustStore");
        transportKeyStoreBean.setTrustStoreType("customTrustStoreType");
        transportKeyStoreBean.setTrustStorePassword("customTrustStorePassword");
        transportKeyStoreBean.afterPropertiesSet();
        assertEquals("customType", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("customUri", System.getProperty("javax.net.ssl.keyStore"));
        assertEquals("customPassword", System.getProperty("javax.net.ssl.keyStorePassword"));
        assertEquals("customTrustStore", System.getProperty("javax.net.ssl.trustStore"));
        assertEquals("customTrustStoreType", System.getProperty("javax.net.ssl.trustStoreType"));
        assertEquals("customTrustStorePassword", System.getProperty("javax.net.ssl.trustStorePassword"));
    }

    @Test
    void testAfterPropertiesSetWithDefaultValues() throws Exception {
        transportKeyStoreBean.afterPropertiesSet();
        assertEquals("pkcs12", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("file:#{ systemProperties['user.home'] }/jnfe.pfx", System.getProperty("javax.net.ssl.keyStore"));
        assertNull(System.getProperty("javax.net.ssl.keyStorePassword"));
        assertNull(System.getProperty("javax.net.ssl.trustStore"));
        assertNull(System.getProperty("javax.net.ssl.trustStoreType"));
        assertNull(System.getProperty("javax.net.ssl.trustStorePassword"));
    }

    @Test
    void testAfterPropertiesSetWithPartialProperties() throws Exception {
        transportKeyStoreBean.setKeyStoreType("customType");
        transportKeyStoreBean.setTrustStore("customTrustStore");
        transportKeyStoreBean.afterPropertiesSet();
        assertEquals("customType", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("file:#{ systemProperties['user.home'] }/jnfe.pfx", System.getProperty("javax.net.ssl.keyStore"));
        assertNull(System.getProperty("javax.net.ssl.keyStorePassword"));
        assertEquals("customTrustStore", System.getProperty("javax.net.ssl.trustStore"));
        assertNull(System.getProperty("javax.net.ssl.trustStoreType"));
        assertNull(System.getProperty("javax.net.ssl.trustStorePassword"));
    }
}
