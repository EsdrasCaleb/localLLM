package br.com.jnfe.base;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Method;
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

class TransportKeyStoreBean_afterPropertiesSet_6_1_Test {

    private TransportKeyStoreBean bean;

    private Logger loggerMock;

    @BeforeEach
    void setUp() {
        loggerMock = Mockito.mock(Logger.class);
        bean = new TransportKeyStoreBean();
        try {
            // Crucial:  Set up the mock logger
            java.lang.reflect.Field loggerField = TransportKeyStoreBean.class.getDeclaredField("logger");
            loggerField.setAccessible(true);
            loggerField.set(bean, loggerMock);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set up mock logger: " + e.getMessage());
        }
    }

    @Test
    void testAfterPropertiesSet_AllNull() throws Exception {
        bean.afterPropertiesSet();
        Mockito.verify(loggerMock, Mockito.times(2)).warn(Mockito.anyString());
        // Verify system properties are set to defaults
        assertEquals("file:#{ systemProperties['user.home'] }/jnfe.pfx", System.getProperty("javax.net.ssl.keyStore"));
        assertEquals("pkcs12", System.getProperty("javax.net.ssl.keyStoreType"));
        assertNull(System.getProperty("javax.net.ssl.keyStorePassword"));
        assertNull(System.getProperty("javax.net.ssl.trustStore"));
        assertNull(System.getProperty("javax.net.ssl.trustStoreType"));
        assertNull(System.getProperty("javax.net.ssl.trustStorePassword"));
    }

    @Test
    void testAfterPropertiesSet_KeyStoreSet() throws Exception {
        bean.setKeyStoreUri("myKeyStoreUri");
        bean.setKeyStoreType("myKeyStoreType");
        bean.setKeyStorePassword("myKeyStorePassword");
        bean.setTrustStore("myTrustStore");
        bean.setTrustStoreType("myTrustStoreType");
        bean.setTrustStorePassword("myTrustStorePassword");
        bean.afterPropertiesSet();
        Mockito.verify(loggerMock, Mockito.times(0)).warn(Mockito.anyString());
        assertEquals("myKeyStoreUri", System.getProperty("javax.net.ssl.keyStore"));
        assertEquals("myKeyStoreType", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("myKeyStorePassword", System.getProperty("javax.net.ssl.keyStorePassword"));
        assertEquals("myTrustStore", System.getProperty("javax.net.ssl.trustStore"));
        assertEquals("myTrustStoreType", System.getProperty("javax.net.ssl.trustStoreType"));
        assertEquals("myTrustStorePassword", System.getProperty("javax.net.ssl.trustStorePassword"));
    }

    @Test
    void testAfterPropertiesSet_KeyStoreTypeNull() throws Exception {
        bean.setKeyStoreUri("myKeyStoreUri");
        bean.setKeyStorePassword("myKeyStorePassword");
        bean.setTrustStore("myTrustStore");
        // Test case for null keyStoreType
        bean.setKeyStoreType(null);
        bean.afterPropertiesSet();
        Mockito.verify(loggerMock).warn(Mockito.contains("Using default keyStoreType."));
        assertEquals("file:#{ systemProperties['user.home'] }/jnfe.pfx", System.getProperty("javax.net.ssl.keyStore"));
        assertEquals("pkcs12", System.getProperty("javax.net.ssl.keyStoreType"));
    }
}
